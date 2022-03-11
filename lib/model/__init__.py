import torch
import torch.nn as nn


class Model(object):
    def __init__(self, model_config, data_config):
        super(Model, self).__init__()

        if model_config['MODEL'] == 'PoseLifter':
            from lib.model.poselifter import PoseLifter
            pos_model = PoseLifter(num_joints=model_config['NUM_KPTS'])
        else:
            raise ValueError('Unrecognized mdoel {}'.format(model_config['MODEL']))

        if torch.cuda.is_available():
            pos_model = nn.DataParallel(pos_model).cuda()

        self.pos_model = pos_model

    def get_pos_model(self):
        return self.pos_model
