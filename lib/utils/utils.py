
import os
import json
import torch
import logging
import hashlib
import numpy as np
from time import strftime


def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    :param func:
    :param args:
    :param unsqueeze:
    :return:
    """

    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result


def deterministic_random(min_value, max_value, data):
    """

    :param min_value:
    :param max_value:
    :param data:
    :return:
    """
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def check_configs(args, data_config, model_config, train_config, plot_config):
    """

    :param data_config:
    :param model_config:
    :param train_config:
    :param plot_config:
    :return:
    """
    valid = True
    e_str = ''

    try:
        if args.resume != '':
            e_str = 'You are training model with [RESUME], [EVALUATE] is supposed to be empty.'
            assert args.evaluate == ''
        if args.evaluate != '':
            e_str = 'You are evaluating model with [EVALUATE], [RESUME] is supposed to be empty.'
            assert args.resume == ''

        e_str = '[CHECKPOINT] is not supposed to be empty.'
        assert train_config['CHECKPOINT'] != ''

        if data_config['INTRINSIC_ENCODING']:
            e_str = 'RAY_ENCODING is supposed to be turned off'
            assert data_config['RAY_ENCODING'] == False

            e_str = 'INTRINSIC_ENCODING requires 2 dimensional input feature'
            assert model_config['INPUT_DIM'] == 2

        if data_config['RAY_ENCODING']:
            e_str = 'INTRINSIC_ENCODING is supposed to be turned off'
            assert data_config['INTRINSIC_ENCODING'] == False

            e_str = 'RAY_ENCODING requires 3 dimensional input feature'
            assert model_config['INPUT_DIM'] == 3

    except AssertionError:
        valid = False

    return valid, e_str


def init_config(args, data_config, model_config, train_config, plot_config):
    """

    :param data_config:
    :param model_config:
    :param train_config:
    :param plot_config:
    :return:
    """
    check_configs(args, data_config, model_config, train_config, plot_config)
    timestamp = args.timestamp
    if args.timestamp == '':
        timestamp = strftime("%b_%d_%Y_%H_%M_%S")
        train_config['CHECKPOINT'] = os.path.join(
            train_config['CHECKPOINT'],
            '{}_{}_{}_{}_FRAME{}_LR{}_EPOCH{}_BATCH{}_{}'.format(train_config['CATEGORY'],data_config['DATASET'],
                                                                 model_config['STAGE'],
                                                                 model_config['MODEL'], model_config['NUM_FRAMES'],
                                                                 train_config['LEARNING_RATE'], train_config['EPOCHS'],
                                                                 train_config['BATCH_SIZE'], timestamp)
        )
    else:
        # assumes that localvars.SAVE_DIR is consistent
        all_exps = [d for d in os.listdir(train_config['CHECKPOINT']) if
                    os.path.isdir(os.path.join(train_config['CHECKPOINT'], d))]
        the_exp = [d for d in all_exps if d[-len(timestamp):] == timestamp][0]

        train_config['CHECKPOINT'] = os.path.join(train_config['CHECKPOINT'], the_exp)

    if not os.path.exists(train_config['CHECKPOINT']):
        os.makedirs(train_config['CHECKPOINT'])

    if args.evaluate != '' or args.resume != '':
        config_dir = os.path.join(train_config['CHECKPOINT'], 'configs')
        p = os.path.join(config_dir, 'train_config.json')
        if os.path.exists(p):
            train_config = json.load(open(p, 'r'))
        else:
            with open(p, 'w') as fp:
                json.dump(train_config, fp, indent=4)

        p = os.path.join(config_dir, 'model_config.json')
        if os.path.exists(p):
            model_config = json.load(open(p, 'r'))
        else:
            with open(p, 'w') as fp:
                json.dump(model_config, fp, indent=4)

        p = os.path.join(config_dir, 'data_config.json')
        if os.path.exists(p):
            data_config = json.load(open(p, 'r'))
        else:
            with open(p, 'w') as fp:
                json.dump(data_config, fp, indent=4)

        p = os.path.join(train_config['CHECKPOINT'], 'configs', 'plot_config.json')
        if os.path.exists(p):
            plot_config = json.load(open(p, 'r'))
        else:
            with open(p, 'w') as fp:
                json.dump(plot_config, fp, indent=4)

        logger = logging.getLogger('EvalLogger')
    else:
        save_str = "rsync -au  --exclude 'data/' --exclude 'checkpoint/' --include '*/' --include '*.py' --exclude '*' . "
        os.system(save_str + train_config['CHECKPOINT'] + "/source")

        # save config dictionaries as json files
        os.makedirs(os.path.join(train_config['CHECKPOINT'], 'configs'))

        p = os.path.join(train_config['CHECKPOINT'], 'configs', 'train_config.json')
        with open(p, 'w') as fp:
            json.dump(train_config, fp, indent=4)

        p = os.path.join(train_config['CHECKPOINT'], 'configs', 'model_config.json')
        with open(p, 'w') as fp:
            json.dump(model_config, fp, indent=4)

        p = os.path.join(train_config['CHECKPOINT'], 'configs', 'data_config.json')
        with open(p, 'w') as fp:
            json.dump(data_config, fp, indent=4)

        p = os.path.join(train_config['CHECKPOINT'], 'configs', 'plot_config.json')
        with open(p, 'w') as fp:
            json.dump(plot_config, fp, indent=4)

        logger = logging.getLogger('TrainLogger')

    logger.setLevel(logging.INFO)
    if len(logger.handlers) > 0:
        logger.handlers = list()
    fh = logging.FileHandler(os.path.join(train_config['CHECKPOINT'], '{}.log'.format(logger.name)))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    print('checkpoint path: {}'.format(train_config['CHECKPOINT']))
    return data_config, model_config, train_config, plot_config, logger
