import torch
import torch.nn as nn


class Model(object):
    def __init__(self, model_config, is_train=True):
        super(Model, self).__init__()

        trj_model = None

        if model_config['MODEL'] == 'PoseFormer':
            from lib.model.poseformer import PoseTransformer
            receptive_field = model_config['NUM_FRAMES']

            if is_train:
                pos_model = PoseTransformer(num_frame=receptive_field, num_joints_in=model_config['NUM_KPTS'],
                                            num_joints_out=model_config['NUM_KPTS'],
                                            in_chans=model_config['INPUT_DIM'],
                                            embed_dim_ratio=32, depth=4,
                                            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                            drop_path_rate=0.1)
                if model_config['TRAJECTORY_MODEL']:
                    trj_model = PoseTransformer(num_frame=receptive_field, num_joints_in=model_config['NUM_KPTS'],
                                                num_joints_out=1,
                                                in_chans=model_config['INPUT_DIM'],
                                                embed_dim_ratio=32, depth=4,
                                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                                drop_path_rate=0.1)

            else:
                pos_model = PoseTransformer(num_frame=receptive_field, num_joints_in=model_config['NUM_KPTS'],
                                            num_joints_out=model_config['NUM_KPTS'],
                                            in_chans=model_config['INPUT_DIM'],
                                            embed_dim_ratio=32, depth=4,
                                            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                            drop_path_rate=0)
                if model_config['TRAJECTORY_MODEL']:
                    trj_model = PoseTransformer(num_frame=receptive_field, num_joints_in=model_config['NUM_KPTS'],
                                                num_joints_out=1,
                                                in_chans=model_config['INPUT_DIM'],
                                                embed_dim_ratio=32, depth=4,
                                                num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                                                drop_path_rate=0)
            # currently poseformer doesn't initialize trajectory model

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