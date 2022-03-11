import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Augmentation Testing script')

    parser.add_argument('--model', default='ray3d', type=str,
                        help='please choose model: 1. ray3d;  2. videopose; 3. rie; 4.poseformer; 5. poselifter ' )

    parser.add_argument('--type', default='Intrinsic', type=str,
                        help='Intrinic')

    parser.add_argument('--device', default='1', type=str, help='')

    parser.add_argument('--c', default='absolute', type=str, help='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # MODEL = 'RAY3D'
    # DEVICE = '0'

    args = parse_args()

    TYPE = args.type 
    MODEL = args.model
    DEVICE = args.device
    C = args.c
    PROJECT_PATH = ''
    MODEL_PATH = ''

    SAVE_LOG_PATH = '/ssd/model_eval_log/' + MODEL
    if not os.path.exists(SAVE_LOG_PATH):
        os.makedirs(SAVE_LOG_PATH)

    if MODEL == 'ray3d':
        # set up model here
        PROJECT_PATH = '/ssd/yzhan/EXP/Ray3D/'
        MODEL_PATH = 'checkpoint/PARALLEL_CAMERA_AUG_h36m_aug_3_RAY3D_FRAME9_LR0.0005_EPOCH20_BATCH32768'
        cfg = 'cfg_ray3d_h36m_stage3'

    os.chdir(PROJECT_PATH)
    
    LOG_FILE = os.path.join(MODEL_PATH, 'EvalLogger.log')
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    evaluate = 'best_epoch.bin'
    dataset = 'h36m_aug'

    # set up data path here
    GT_3D_PATH = '/ssd/yzhan/data/benchmark/3D/h36m/annotations/gt/data_3d_h36m.npz'  # h36m 3D GT
    GT_2D_PATH = '/ssd/ray3d/camera.intrinsic/npz/'  # synthetic 2D GT
    CAM_PARAM_PATH = '/ssd/ray3d/camera.intrinsic/json/'

    TRANSLATION = 2.0
    YAW = 0
    PITCH = 0
    CAM = 1

    SUBJECTS = ['S9,S11']

    FBIAS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    CBIAS = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40]
    TEMP = 'TRANSLATION{}_YAW{}_PITCH{}_CAM{}_FBAIS{}_CBIAS{}'

    for fbias in FBIAS:
        for cbias in CBIAS:
            filename = TEMP.format(TRANSLATION, YAW, PITCH, CAM, fbias, cbias)
            gt_2d_file = os.path.join(GT_2D_PATH, filename + '.npz')
            cam_param_file = os.path.join(CAM_PARAM_PATH, filename + '.json')

            if not os.path.exists(gt_2d_file) or not os.path.exists(cam_param_file):
                import ipdb
                ipdb.set_trace()

            data_config = json.load(open(os.path.join(MODEL_PATH, 'configs', 'data_config.json'), 'r'))
            data_config['DATASET'] = dataset
            data_config['GT_3D'] = GT_3D_PATH
            data_config['GT_2D'] = gt_2d_file
            data_config['CAMERA_PARAM'] = cam_param_file
            data_config['CAMERA_WISE_PERFORMANCE'] = True
            data_config['TRAIN_SUBJECTS'] = 'S1,S5,S6,S7,S8'

            train_config = json.load(open(os.path.join(MODEL_PATH, 'configs', 'train_config.json'), 'r'))
            train_config['DEVICE'] = DEVICE
            train_config['TEST_TIME_AUGMENTATION'] = True
            json.dump(train_config, open(os.path.join(MODEL_PATH, 'configs', 'train_config.json'), 'w'), indent=4)
    
            model_config = json.load(open(os.path.join(MODEL_PATH, 'configs', 'model_config.json'), 'r'))
            model_config['TRAJECTORY_MODEL'] = True
            json.dump(model_config, open(os.path.join(MODEL_PATH, 'configs', 'model_config.json'), 'w'), indent=4)

            for subject in SUBJECTS:
                data_config['TEST_SUBJECTS'] = subject
                json.dump(data_config, open(os.path.join(MODEL_PATH, 'configs', 'data_config.json'), 'w'), indent=4)

                cmd = 'python3 main.py --cfg {} --evaluate {} --timestamp {}'.format(cfg, evaluate, os.path.basename(MODEL_PATH))
                os.system(cmd)

    if os.path.exists(LOG_FILE):
        os.system('cp {} {}'.format(LOG_FILE, os.path.join(SAVE_LOG_PATH, TYPE + '.' +C+  '.log')))