import torch
import torch.nn as nn


class Model(object):
    def __init__(self, model_config, data_config, is_train=True):
        super(Model, self).__init__()

        trj_model = None

        if model_config['CAMERA_EMBDDING']:
            extrinsic_dim = model_config['EXTRINSIC_DIM']
            embedd_dim = model_config['EMBEDD_DIM']
        else:
            extrinsic_dim = 0
            embedd_dim = 0

        if model_config['MODEL'] == 'RIE':
            from lib.model.rie import RIEModel
            filter_widths = [int(x) for x in model_config['ARCHITECTURE'].split(',')]

            num_joints_in = model_config['NUM_KPTS']
            num_joints_out = model_config['NUM_KPTS']
            from lib.model.rie import RIEModel
            pos_model = RIEModel(num_joints_in, model_config['INPUT_DIM'],
                                 num_joints_out,
                                 filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                 dropout=model_config['DROPOUT'],
                                 channels=model_config['CHANNELS'], latten_features=model_config['LATENT_FEATURES_DIM'],
                                 dense=model_config['DENSE'], is_train=is_train,
                                 Optimize1f=not model_config['DISABLE_OPTIMIZATIONS'], stage=model_config['STAGE'],
                                 extrinsic_dim=extrinsic_dim,
                                 embedd_dim=embedd_dim)
            if model_config['TRAJECTORY_MODEL']:
                from lib.model.rie import RIETrajectoryModel
                trj_model = RIETrajectoryModel(num_joints_in, model_config['INPUT_DIM'],
                                               num_joints_out,
                                               filter_widths=filter_widths, causal=model_config['CAUSAL'],
                                               dropout=model_config['DROPOUT'],
                                               channels=model_config['CHANNELS'],
                                               latten_features=model_config['LATENT_FEATURES_DIM'],
                                               dense=model_config['DENSE'], is_train=is_train,
                                               Optimize1f=not model_config['DISABLE_OPTIMIZATIONS'],
                                               stage=model_config['STAGE'],
                                               extrinsic_dim=extrinsic_dim,
                                               embedd_dim=embedd_dim)

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
