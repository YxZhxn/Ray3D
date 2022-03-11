import os
import importlib
from time import time

import torch
import torch.optim as optim

from lib.model import Model
from lib.dataset import Data
from lib.utils.utils import init_config, load_weight
from lib.train_val.trainer import Trainer
from lib.dataloader.generators import ChunkedGenerator, UnchunkedGenerator
from lib.visualization.plotter import ExperimentPlotter
import numpy as np
import random
import torch.backends.cudnn as cudnn


def main():
    from cfg.arguments import parse_args
    args = parse_args()

    """-------- random seed  --------"""
    random_seed = args.random_seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    # copy from #https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True


    cfg = importlib.import_module('cfg.' + args.cfg)
    data_config, model_config, train_config, plot_config = \
        cfg.data_config, cfg.model_config, cfg.train_config, cfg.plot_config

    """-------- parse arguments --------"""
    data_config, model_config, train_config, plot_config, mlog = init_config(args, data_config, model_config,
                                                                             train_config, plot_config)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = train_config['DEVICE']

    """-------- output mode --------"""
    if args.render:
        mlog.info('MODE: RENDER')
        plotter = None
    elif args.evaluate != '':
        mlog.info('MODE: EVALUATE')
        plotter = None
    elif args.resume != '':
        mlog.info('MODE: RESUME')
        plotter = None
    else:
        mlog.info('MODE:TRAINING')
        if plot_config['EXP_PLOTTING']:
            env = os.path.basename(train_config['CHECKPOINT'])
            plotter = ExperimentPlotter(data_config, model_config, train_config, plot_config, env)
        else:
            plotter = None

    """-------- load dataset --------"""
    mlog.info('Loading dataset: {}'.format(data_config['DATASET']))
    mlog.info(data_config['GT_2D'])
    pose_data = Data(data_config)
    kps_left, kps_right = pose_data.get_2d_kpts()
    joints_left, joints_right = pose_data.get_3d_joints()

    """-------- loader for train and test --------"""
    subjects_train = data_config['TRAIN_SUBJECTS'].split(',')
    subjects_test = data_config['TEST_SUBJECTS'].split(',')

    action_filter = None if data_config['ACTIONS'] == '*' else data_config['ACTIONS'].split(',')
    if action_filter is not None:
        mlog.info('Selected actions: {}'.format(action_filter))

    cameras_train, poses_train, poses_train_2d = pose_data.fetch_via_subject(subjects_train, action_filter,
                                                                             subset=data_config['SUBSET'])
    cameras_valid, poses_valid, poses_valid_2d = pose_data.fetch_via_subject(subjects_test, action_filter)

    receptive_field = model_config['NUM_FRAMES']
    mlog.info('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    if model_config['CAUSAL']:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    train_generator = ChunkedGenerator(train_config['BATCH_SIZE'] // data_config['STRIDE'],
                                       cameras_train, poses_train, poses_train_2d,
                                       data_config['STRIDE'],
                                       pad=pad, causal_shift=causal_shift, shuffle=True,
                                       augment=train_config['TRAIN_TIME_AUGMENTATION'],
                                       kps_left=kps_left, kps_right=kps_right,
                                       joints_left=joints_left, joints_right=joints_right)
    mlog.info('INFO: Training on {} frames'.format(train_generator.num_frames()))

    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right,
                                        joints_left=joints_left, joints_right=joints_right)
    mlog.info('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    """-------- model --------"""
    mlog.info('INFO: Init model:' + model_config['MODEL'])
    assert model_config['NUM_KPTS'] == pose_data.keypoints_metadata['num_joints']
    train_delegator = Model(model_config, data_config, is_train=True)
    test_delegator = Model(model_config, data_config, is_train=False)
    models = {
        'train_pos': train_delegator.get_pos_model(),
        'train_trj': train_delegator.get_trj_model(),
        'test_pos': test_delegator.get_pos_model(),
        'test_trj': test_delegator.get_trj_model()
    }
    model_params = 0
    for parameter in models['train_pos'].parameters():
        model_params += parameter.numel()
    if not models['train_trj'] is None:
        for parameter in models['train_trj'].parameters():
            model_params += parameter.numel()
    mlog.info('INFO: Trainable parameter count: {}'.format(model_params))

    trainable_params = list()
    trainable_params.append(
        {'name': 'pos_model', 'params': models['train_pos'].parameters(), 'lr': train_config['LEARNING_RATE']}
    )
    if not models['train_trj'] is None:
        trainable_params.append(
            {'name': 'trj_model', 'params': models['train_trj'].parameters(), 'lr': train_config['LEARNING_RATE_TRAJECTORY']}
        )
    if train_config['OPTIMIZER'] == 'AdamW':
        optimizer = optim.AdamW(
            trainable_params,
            lr=train_config['LEARNING_RATE'],
            weight_decay=0.1
        )
    elif train_config['OPTIMIZER'] == 'Adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=train_config['LEARNING_RATE'],
            amsgrad=True
        )
    else:
        raise ValueError('Unsupported optimizer: {}'.format(train_config['OPTIMIZER']))

    epoch = 1
    if model_config['PRETRAIN'] != '':
        mlog.info('Loading pretrained model: {}'.format(model_config['PRETRAIN']))
        checkpoint = torch.load(
            model_config['PRETRAIN'],
            map_location=lambda storage, loc: storage
        )
        pretrain_dict = checkpoint['model_pos']
        model_dict = models['train_pos'].state_dict()
        state_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict.keys() and 'Integration' not in k}
        model_dict.update(state_dict)
        # models['train_pos'].load_state_dict(model_dict, strict=True)
        # models['test_pos'].load_state_dict(model_dict, strict=True)
        load_weight(models['train_pos'], model_dict)
        load_weight(models['test_pos'], model_dict)
        cnt = 0
        for name, value in models['train_pos'].named_parameters():
            # 243 -> 167, 9 -> 77, 27 -> 107, 81 -> 137
            if cnt < 77:
                value.requires_grad = False
            cnt = cnt + 1

        if 'model_trj' in checkpoint and model_config['TRAJECTORY_MODEL']:
            # models['train_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            # models['test_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            load_weight(models['train_trj'], checkpoint['model_trj'])
            load_weight(models['test_trj'], checkpoint['model_trj'])
    if model_config['FINETUNE'] != '':
        mlog.info('Loading pretrained model: {}'.format(model_config['FINETUNE']))
        checkpoint = torch.load(
            model_config['FINETUNE'],
            map_location=lambda storage, loc: storage
        )
        # models['train_pos'].load_state_dict(checkpoint['model_pos'], strict=True)
        # models['test_pos'].load_state_dict(checkpoint['model_pos'], strict=True)
        load_weight(models['train_pos'], checkpoint['model_pos'])
        load_weight(models['test_pos'], checkpoint['model_pos'])
        if 'model_trj' in checkpoint and model_config['TRAJECTORY_MODEL']:
            # models['train_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            # models['test_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            load_weight(models['train_trj'], checkpoint['model_trj'])
            load_weight(models['test_trj'], checkpoint['model_trj'])
    best_performance = None
    if args.resume or args.evaluate:
        chk_filename = os.path.join(train_config['CHECKPOINT'],
                                    args.resume if args.resume else args.evaluate)
        mlog.info('Loading checkpoint: {}'.format(chk_filename))
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        # models['train_pos'].load_state_dict(checkpoint['model_pos'], strict=True)
        # models['test_pos'].load_state_dict(checkpoint['model_pos'], strict=True)
        load_weight(models['train_pos'], checkpoint['model_pos'])
        load_weight(models['test_pos'], checkpoint['model_pos'])
        if 'model_trj' in checkpoint and model_config['TRAJECTORY_MODEL']:
            # models['train_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            # models['test_trj'].load_state_dict(checkpoint['model_trj'], strict=True)
            load_weight(models['train_trj'], checkpoint['model_trj'])
            load_weight(models['test_trj'], checkpoint['model_trj'])

        if args.resume:
            epoch = checkpoint['epoch']
            best_performance = checkpoint['best_performance']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
                train_generator.set_random_state(checkpoint['random_state'])
            else:
                mlog.info(
                    'WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            train_config['LEARNING_RATE'] = checkpoint['lr']

    trainval_engine = Trainer(data_config, model_config, train_config, plot_config,
                              train_generator, test_generator,
                              models, optimizer,
                              kps_left, kps_right, joints_left, joints_right, plotter, best_performance)

    """-------- training --------"""
    if not args.evaluate:
        mlog.info('Training from epoch: {}'.format(epoch))
        mlog.info('** Note: reported losses are averaged over all frames.')
        mlog.info('** The final evaluation will be carried out after the last training epoch.')

        while epoch <= train_config['EPOCHS']:
            start_time = time()

            losses_3d_train, current_lr = trainval_engine.train(epoch, mlog)

            losses_3d_eval = trainval_engine.test(epoch, mlog)

            if epoch % 16 == 0:
                trainval_engine.evaluate(mlog, subjects_test, pose_data, action_filter, pad, causal_shift, epoch, plot=True)

            if plot_config['EXP_PLOTTING'] and plot_config['SAVE_PLOTS'] and plotter:
                plotter.save_env()

            elapsed = (time() - start_time) / 60

            mlog.info('[%d] time %.2f lr %f 3d_train %f 3d_eval %f ' % (
                epoch,
                elapsed,
                current_lr,
                losses_3d_train * 1000,
                losses_3d_eval * 1000
            ))
            epoch += 1

    if args.render:
        mlog.info('RENDERING...')
        keypoints = pose_data.get_keypoints()
        dataset = pose_data.get_dataset()
        file_names = pose_data.file_names
        trainval_engine.render(dataset, keypoints, pose_data.keypoints_metadata, pad, causal_shift, kps_left, kps_right,
                               joints_left, joints_right, file_names)
    else:
        """-------- testing --------"""
        mlog.info('Evaluating...')
        trainval_engine.evaluate(mlog, subjects_test, pose_data, action_filter, pad, causal_shift, epoch, plot=False)


if __name__ == '__main__':
    # fix random
    main()
