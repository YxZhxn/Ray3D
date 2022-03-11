import torch
import torch.nn as nn


class Model(object):
    def __init__(self, model_config, data_config, is_train=True):
        super(Model, self).__init__()

        trj_model = None

        if model_config['MODEL'] == 'VideoPose3D':
            from lib.model.videopose3D import TemporalModel, TemporalModelOptimized1f

            filter_widths = [int(x) for x in model_config['ARCHITECTURE'].split(',')]

            num_joints_in = model_config['NUM_KPTS']
            num_joints_out = model_config['NUM_KPTS']

            if data_config['ADD_HEIGHT']:
                num_joints_in += 1

            if is_train and not model_config['DISABLE_OPTIMIZATIONS'] and not model_config['DENSE']:
                # Use optimized model for single-frame predictions
                pos_model = TemporalModelOptimized1f(num_joints_in, model_config['INPUT_DIM'],
                                                     num_joints_out,
                                                     filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                                     dropout=model_config['DROPOUT'],
                                                     channels=model_config['CHANNELS'])
                if model_config['TRAJECTORY_MODEL']:
                    trj_model = TemporalModelOptimized1f(num_joints_in, model_config['INPUT_DIM'],
                                                         1,
                                                         filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                                         dropout=model_config['DROPOUT'],
                                                         channels=model_config['CHANNELS'])
            else:
                pos_model = TemporalModel(num_joints_in, model_config['INPUT_DIM'],
                                          num_joints_out,
                                          filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                          dropout=model_config['DROPOUT'],
                                          channels=model_config['CHANNELS'],
                                          dense=model_config['DENSE'])
                if model_config['TRAJECTORY_MODEL']:
                    trj_model = TemporalModel(num_joints_in, model_config['INPUT_DIM'],
                                              1,
                                              filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                              dropout=model_config['DROPOUT'],
                                              channels=model_config['CHANNELS'],
                                              dense=model_config['DENSE'])

        else:
            raise ValueError('Unrecognized mdoel {}'.format(model_config['MODEL']))

        if torch.cuda.is_available():
            pos_model = nn.DataParallel(pos_model).cuda()
            trj_model = nn.DataParallel(trj_model).cuda() if not trj_model is None else None

        self.pos_model = pos_model
        self.trj_model = trj_model

    def get_pos_model(self):
        return self.pos_model

    def get_trj_model(self):
        return self.trj_model
