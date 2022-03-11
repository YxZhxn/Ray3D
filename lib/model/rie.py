
import torch
import torch.nn as nn
from lib.model.embedding import Embedding


class TemporalBlock(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, channels=1024, latten_features=256, dense=False,
                 is_train=True, Optimize1f=True):
        super().__init__()

        self.is_train = is_train
        self.augment = False
        self.Optimize1f = Optimize1f
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        # self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)
        self.shrink = nn.Conv1d(channels, latten_features, 1)

        if self.Optimize1f == False:
            self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)
        else:
            self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0],
                                         stride=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            if self.Optimize1f == False:
                layers_conv.append(nn.Conv1d(channels, channels,
                                             filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                                             dilation=next_dilation if not dense else 1,
                                             bias=False))
            else:
                layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def set_training_status(self, is_train):
        self.is_train = is_train

    def set_augment(self, augment):
        self.augment = augment

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def forward(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            if self.Optimize1f == False:
                res = x[:, :, pad + shift: x.shape[2] - pad + shift]
            else:
                res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2:: self.filter_widths[i + 1]]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        x = x.permute(0, 2, 1)

        x_sz = x.shape
        x = x.reshape(x_sz[0] * x_sz[1], x_sz[2]).unsqueeze(1)

        return x


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.25):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


class FCBlock(nn.Module):

    def __init__(self, channel_in, channel_out, linear_size, block_num):
        super(FCBlock, self).__init__()

        self.linear_size = linear_size
        self.block_num = block_num
        self.layers = []
        self.channel_in = channel_in
        self.stage_num = 3
        self.p_dropout = 0.25
        self.fc_1 = nn.Linear(self.channel_in, self.linear_size)
        self.bn_1 = nn.BatchNorm1d(self.linear_size)
        for i in range(block_num):
            self.layers.append(Linear(self.linear_size, self.p_dropout))
        self.fc_2 = nn.Linear(self.linear_size, channel_out)

        self.layers = nn.ModuleList(self.layers)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        x = self.fc_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        for i in range(self.block_num):
            x = self.layers[i](x)
        x = self.fc_2(x)

        return x


class RIEModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, latten_features=256,
                 channels=1024, dense=False, is_train=True, Optimize1f=True, stage=1,
                 extrinsic_dim=12, embedd_dim=64):
        super(RIEModel, self).__init__()
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        is_train -- if the model runs in training mode or not
        Optimize1f=True -- using 1 frame optimization or not
        stage -- current stage when using the multi-stage optimization method
        """
        self.augment = False
        self.is_train = is_train
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.in_features = in_features
        self.latten_features = latten_features
        self.stage = stage
        if self.num_joints_in == 17:
            self.LocalLayer_Torso = TemporalBlock(5 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                channels, self.latten_features, dense, is_train, Optimize1f)
        elif self.num_joints_in == 15:
            self.LocalLayer_Torso = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                  channels, self.latten_features, dense, is_train, Optimize1f)
        elif self.num_joints_in == 14:
            self.LocalLayer_Torso = TemporalBlock(2 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                  channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_LArm = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_RArm = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_LLeg = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)
        self.LocalLayer_RLeg = TemporalBlock(3 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                             channels, self.latten_features, dense, is_train, Optimize1f)

        self.pad = (self.receptive_field() - 1) // 2

        self.GlobalInfo = FCBlock(num_joints_in * self.in_features, self.latten_features, 1024, 2)

        if stage != 1:
            self.FuseBlocks = nn.ModuleList([])
            for i in range(5):
                self.FuseBlocks.append(
                    FCBlock(self.latten_features * 4, self.latten_features, 1024, 1)
                )

        self.camera_embedding = True if (extrinsic_dim > 0 and embedd_dim > 0) else False
        self.extrinsic_dim = extrinsic_dim
        self.embedd_dim = embedd_dim
        if self.camera_embedding:
            self.embedder = Embedding(in_channels=extrinsic_dim, out_channels=embedd_dim)

        self.out_features_dim = self.latten_features * 2 if stage == 1 else self.latten_features * 3
        self.out_features_dim += embedd_dim

        if self.num_joints_in == 17:
            self.Integration_Torso = FCBlock(self.out_features_dim, 5 * 3, 1024, 1)
        elif self.num_joints_in == 15:
            self.Integration_Torso = FCBlock(self.out_features_dim, 3 * 3, 1024, 1)
        elif self.num_joints_in == 14:
            self.Integration_Torso = FCBlock(self.out_features_dim, 2 * 3, 1024, 1)
        self.Integration_LArm = FCBlock(self.out_features_dim, 3 * 3, 1024, 1)
        self.Integration_RArm = FCBlock(self.out_features_dim, 3 * 3, 1024, 1)
        self.Integration_LLeg = FCBlock(self.out_features_dim, 3 * 3, 1024, 1)
        self.Integration_RLeg = FCBlock(self.out_features_dim, 3 * 3, 1024, 1)

    def set_bn_momentum(self, momentum):
        self.LocalLayer_Torso.set_bn_momentum(momentum)
        self.LocalLayer_LArm.set_bn_momentum(momentum)
        self.LocalLayer_RArm.set_bn_momentum(momentum)
        self.LocalLayer_LLeg.set_bn_momentum(momentum)
        self.LocalLayer_RLeg.set_bn_momentum(momentum)

    def set_training_status(self, is_train):
        self.is_train = is_train
        self.LocalLayer_Torso.set_training_status(is_train)
        self.LocalLayer_LArm.set_training_status(is_train)
        self.LocalLayer_RArm.set_training_status(is_train)
        self.LocalLayer_LLeg.set_training_status(is_train)
        self.LocalLayer_RLeg.set_training_status(is_train)

    def set_augment(self, augment):
        self.augment = augment
        self.LocalLayer_Torso.set_augment(augment)
        self.LocalLayer_LArm.set_augment(augment)
        self.LocalLayer_RArm.set_augment(augment)
        self.LocalLayer_LLeg.set_augment(augment)
        self.LocalLayer_RLeg.set_augment(augment)

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        return self.LocalLayer_Torso.receptive_field()

    def forward(self, x, param):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        pad = (self.receptive_field() - 1) // 2
        in_current = x[:, x.shape[1] // self.in_features:x.shape[1] // self.in_features + 1]

        in_current = in_current.reshape(in_current.shape[0] * in_current.shape[1], -1)

        x_sz = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        sz = x.shape

        # Positional information encoding
        diff = x - x[:, 0:self.in_features, :].repeat(1, sz[1] // self.in_features, 1)

        # Temporal information encoding
        diff_t = x - x[:, :, x.shape[2] // self.in_features:x.shape[2] // self.in_features + 1].expand(sz[0], sz[1], sz[2])

        if self.in_features == 3:
            # Grouping
            if self.num_joints_in == 17:
                in_Torso = torch.cat(
                    (x[:, 0:3, :], x[:, 21:33, :], diff[:, 0:3, :], diff[:, 21:33, :], diff_t[:, 0:3, :], diff_t[:, 21:33, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 42:51, :], diff[:, 42:51, :], diff_t[:, 42:51, :]), dim=1)
                in_RArm = torch.cat((x[:, 33:42, :], diff[:, 33:42, :], diff_t[:, 33:42, :]), dim=1)
                in_LLeg = torch.cat((x[:, 3:12, :], diff[:, 3:12, :], diff_t[:, 3:12, :]), dim=1)
                in_RLeg = torch.cat((x[:, 12:21, :], diff[:, 12:21, :], diff_t[:, 12:21, :]), dim=1)
            elif self.num_joints_in == 15:
                in_Torso = torch.cat(
                    (x[:, 0:6, :], x[:, 42:45, :], diff[:, 0:6, :], diff[:, 42:45, :], diff_t[:, 0:6, :], diff_t[:, 42:45, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 6:15, :], diff[:, 6:15, :], diff_t[:, 6:15, :]), dim=1)
                in_RArm = torch.cat((x[:, 15:24, :], diff[:, 15:24, :], diff_t[:, 15:24, :]), dim=1)
                in_LLeg = torch.cat((x[:, 24:33, :], diff[:, 24:33, :], diff_t[:, 24:33, :]), dim=1)
                in_RLeg = torch.cat((x[:, 33:42, :], diff[:, 33:42, :], diff_t[:, 33:42, :]), dim=1)
            elif self.num_joints_in == 14:
                in_Torso = torch.cat(
                    (x[:, 0:3, :], x[:, 21:24, :], diff[:, 0:3, :], diff[:, 21:24, :], diff_t[:, 0:3, :], diff_t[:, 21:24, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 24:33, :], diff[:, 24:33, :], diff_t[:, 24:33, :]), dim=1)
                in_RArm = torch.cat((x[:, 33:42, :], diff[:, 33:42, :], diff_t[:, 33:42, :]), dim=1)
                in_LLeg = torch.cat((x[:, 12:21, :], diff[:, 12:21, :], diff_t[:, 12:21, :]), dim=1)
                in_RLeg = torch.cat((x[:, 3:12, :], diff[:, 3:12, :], diff_t[:, 3:12, :]), dim=1)

        if self.in_features == 2:
            if self.num_joints_in == 17:
                in_Torso = torch.cat(
                    (x[:, 0:2, :], x[:, 14:22, :], diff[:, 0:2, :], diff[:, 14:22, :], diff_t[:, 0:2, :], diff_t[:, 14:22, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 28:34, :], diff[:, 28:34, :], diff_t[:, 28:34, :]), dim=1)
                in_RArm = torch.cat((x[:, 22:28, :], diff[:, 22:28, :], diff_t[:, 22:28, :]), dim=1)
                in_LLeg = torch.cat((x[:, 2:8, :], diff[:, 2:8, :], diff_t[:, 2:8, :]), dim=1)
                in_RLeg = torch.cat((x[:, 8:14, :], diff[:, 8:14, :], diff_t[:, 8:14, :]), dim=1)
            elif self.num_joints_in == 15:
                in_Torso = torch.cat(
                    (x[:, 0:4, :], x[:, 28:30, :], diff[:, 0:4, :], diff[:, 28:30, :], diff_t[:, 0:4, :], diff_t[:, 28:30, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 4:10, :], diff[:, 4:10, :], diff_t[:, 4:10, :]), dim=1)
                in_RArm = torch.cat((x[:, 10:16, :], diff[:, 10:16, :], diff_t[:, 10:16, :]), dim=1)
                in_LLeg = torch.cat((x[:, 16:22, :], diff[:, 16:22, :], diff_t[:, 16:22, :]), dim=1)
                in_RLeg = torch.cat((x[:, 22:28, :], diff[:, 22:28, :], diff_t[:, 22:28, :]), dim=1)
            elif self.num_joints_in == 14:
                in_Torso = torch.cat(
                    (x[:, 0:2, :], x[:, 14:16, :], diff[:, 0:2, :], diff[:, 14:16, :], diff_t[:, 0:2, :], diff_t[:, 14:16, :]),
                    dim=1)
                in_LArm = torch.cat((x[:, 16:22, :], diff[:, 16:22, :], diff_t[:, 16:22, :]), dim=1)
                in_RArm = torch.cat((x[:, 22:28, :], diff[:, 22:28, :], diff_t[:, 22:28, :]), dim=1)
                in_LLeg = torch.cat((x[:, 8:14, :], diff[:, 8:14, :], diff_t[:, 8:14, :]), dim=1)
                in_RLeg = torch.cat((x[:, 2:8, :], diff[:, 2:8, :], diff_t[:, 2:8, :]), dim=1)



        # Global Feature Encoder
        x_global = self.GlobalInfo(in_current)

        # Local Feature Encoder
        xTorso = self.LocalLayer_Torso(in_Torso)
        xLArm = self.LocalLayer_LArm(in_LArm)
        xRArm = self.LocalLayer_RArm(in_RArm)
        xLLeg = self.LocalLayer_LLeg(in_LLeg)
        xRLeg = self.LocalLayer_RLeg(in_RLeg)

        tmp = torch.cat((xTorso, xLArm, xRArm, xLLeg, xRLeg), dim=1)

        if self.stage == 1:
            if self.camera_embedding:
                extrinsic_embedding = self.embedder(param)
                xTorso = torch.cat((tmp[:, 0], x_global, extrinsic_embedding), dim=1)
                xLArm = torch.cat((tmp[:, 1], x_global, extrinsic_embedding), dim=1)
                xRArm = torch.cat((tmp[:, 2], x_global, extrinsic_embedding), dim=1)
                xLLeg = torch.cat((tmp[:, 3], x_global, extrinsic_embedding), dim=1)
                xRLeg = torch.cat((tmp[:, 4], x_global, extrinsic_embedding), dim=1)
            else:
                xTorso = torch.cat((tmp[:, 0], x_global), dim=1)
                xLArm = torch.cat((tmp[:, 1], x_global), dim=1)
                xRArm = torch.cat((tmp[:, 2], x_global), dim=1)
                xLLeg = torch.cat((tmp[:, 3], x_global), dim=1)
                xRLeg = torch.cat((tmp[:, 4], x_global), dim=1)

        else:
            # Feature Fusion
            mix_features = torch.zeros(tmp.shape[0], 5, self.latten_features).to(tmp.device)

            for i, fb in enumerate(self.FuseBlocks):
                mix_features[:, i] = fb(torch.cat((tmp[:, :i, :], tmp[:, (i + 1):, :]), dim=1).reshape(tmp.shape[0],
                                                                                                       self.latten_features * 4))
            if self.camera_embedding:
                extrinsic_embedding = self.embedder(param)
                xTorso = torch.cat((tmp[:, 0], mix_features[:, 0], x_global, extrinsic_embedding), dim=1)
                xLArm = torch.cat((tmp[:, 1], mix_features[:, 1], x_global, extrinsic_embedding), dim=1)
                xRArm = torch.cat((tmp[:, 2], mix_features[:, 2], x_global, extrinsic_embedding), dim=1)
                xLLeg = torch.cat((tmp[:, 3], mix_features[:, 3], x_global, extrinsic_embedding), dim=1)
                xRLeg = torch.cat((tmp[:, 4], mix_features[:, 4], x_global, extrinsic_embedding), dim=1)
            else:
                xTorso = torch.cat((tmp[:, 0], mix_features[:, 0], x_global), dim=1)
                xLArm = torch.cat((tmp[:, 1], mix_features[:, 1], x_global), dim=1)
                xRArm = torch.cat((tmp[:, 2], mix_features[:, 2], x_global), dim=1)
                xLLeg = torch.cat((tmp[:, 3], mix_features[:, 3], x_global), dim=1)
                xRLeg = torch.cat((tmp[:, 4], mix_features[:, 4], x_global), dim=1)

        # Decoder
        xTorso = self.Integration_Torso(xTorso)
        xLArm = self.Integration_LArm(xLArm)
        xRArm = self.Integration_RArm(xRArm)
        xLLeg = self.Integration_LLeg(xLLeg)
        xRLeg = self.Integration_RLeg(xRLeg)
        if self.num_joints_in == 17:
            xTorso = xTorso.view(xTorso.size(0), 5, 3)
        elif self.num_joints_in == 15:
            xTorso = xTorso.view(xTorso.size(0), 3, 3)
        elif self.num_joints_in == 14:
            xTorso = xTorso.view(xTorso.size(0), 2, 3)
        xLArm = xLArm.view(xLArm.size(0), 3, 3)
        xRArm = xRArm.view(xRArm.size(0), 3, 3)
        xLLeg = xLLeg.view(xLLeg.size(0), 3, 3)
        xRLeg = xRLeg.view(xRLeg.size(0), 3, 3)

        if self.num_joints_in == 17:
            x = torch.cat((xTorso[:, 0:1], xLLeg, xRLeg, xTorso[:, 1:5], xRArm, xLArm), dim=1)
        elif self.num_joints_in == 15:
            x = torch.cat((xTorso[:, 0:2], xLLeg, xRLeg, xRArm, xLArm, xTorso[:, 2:3]), dim=1)
        elif self.num_joints_in == 14:
            x = torch.cat((xTorso[:, 0:1], xLLeg, xRLeg, xRArm, xLArm, xTorso[:, 1:2]), dim=1)
        x = x.view(x_sz[0], x_sz[1] - 2 * pad, self.num_joints_in, 3)

        return x


class RIETrajectoryModel(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.2, latten_features=256,
                 channels=1024, dense=False, is_train=True, Optimize1f=True, stage=1,
                 extrinsic_dim=12, embedd_dim=64):
        super(RIETrajectoryModel, self).__init__()
        """
        Initialize this model.

        Arguments:
        num_joints_in -- number of input joints (e.g. 17 for Human3.6M)
        in_features -- number of input features for each joint (typically 2 for 2D input)
        num_joints_out -- number of output joints (can be different than input)
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        is_train -- if the model runs in training mode or not
        Optimize1f=True -- using 1 frame optimization or not
        stage -- current stage when using the multi-stage optimization method
        """
        self.augment = False
        self.is_train = is_train
        self.num_joints_in = num_joints_in
        self.num_joints_out = num_joints_out
        self.in_features = in_features
        self.latten_features = latten_features
        self.stage = stage
        if self.num_joints_in == 17:
            self.LocalLayer = TemporalBlock(17 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                channels, self.latten_features, dense, is_train, Optimize1f)
        elif self.num_joints_in == 15:
            self.LocalLayer = TemporalBlock(15 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                  channels, self.latten_features, dense, is_train, Optimize1f)
        elif self.num_joints_in == 14:
            self.LocalLayer = TemporalBlock(14 * 3, in_features, num_joints_out, filter_widths, causal, dropout,
                                                  channels, self.latten_features, dense, is_train, Optimize1f)

        self.pad = (self.receptive_field() - 1) // 2

        self.GlobalInfo = FCBlock(num_joints_in * self.in_features, self.latten_features, 1024, 2)

        self.camera_embedding = True if (extrinsic_dim > 0 and embedd_dim > 0) else False
        self.extrinsic_dim = extrinsic_dim
        self.embedd_dim = embedd_dim
        if self.camera_embedding:
            self.embedder = Embedding(in_channels=extrinsic_dim, out_channels=embedd_dim)

        self.out_features_dim = self.latten_features * 2
        self.out_features_dim += embedd_dim

        self.Integration = FCBlock(self.out_features_dim, 1 * 3, 1024, 1)


    def set_bn_momentum(self, momentum):
        self.LocalLayer.set_bn_momentum(momentum)


    def set_training_status(self, is_train):
        self.is_train = is_train
        self.LocalLayer.set_training_status(is_train)


    def set_augment(self, augment):
        self.augment = augment
        self.LocalLayer.set_augment(augment)


    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        return self.LocalLayer.receptive_field()


    def forward(self, x, param):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        pad = (self.receptive_field() - 1) // 2
        in_current = x[:, x.shape[1] // self.in_features:x.shape[1] // self.in_features + 1]

        in_current = in_current.reshape(in_current.shape[0] * in_current.shape[1], -1)

        x_sz = x.shape
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        sz = x.shape

        # Positional information encoding
        diff = x - x[:, 0:self.in_features, :].repeat(1, sz[1] // self.in_features, 1)

        # Temporal information encoding
        diff_t = x - x[:, :, x.shape[2] // self.in_features:x.shape[2] // self.in_features + 1].expand(sz[0], sz[1], sz[2])

        x_local = torch.cat((x, diff, diff_t), dim=1)

        # Global Feature Encoder
        x_global = self.GlobalInfo(in_current)

        # Local Feature Encoder
        x_local = self.LocalLayer(x_local)

        if self.camera_embedding:
            extrinsic_embedding = self.embedder(param)
            output = torch.cat((x_local[:, 0], x_global, extrinsic_embedding), dim=1)
        else:
            output = torch.cat((x_local[:, 0], x_global), dim=1)

        # Decoder
        output = self.Integration(output)

        output = output.view(x_sz[0], x_sz[1] - 2 * pad, 1, 3)

        return output