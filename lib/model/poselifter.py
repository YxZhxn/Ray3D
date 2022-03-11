"""
Reference: https://github.com/juyongchang/PoseLifter
"""
import torch
import torch.nn as nn


class ResNetModule(nn.Module):
    def __init__(self, num_features):
        super(ResNetModule, self).__init__()

        modules = []
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))
        modules.append(nn.BatchNorm1d(num_features))
        modules.append(nn.Dropout(0.5))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(num_features, num_features))

        # set weights
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.submod = nn.Sequential(*modules)

    def forward(self, x):
        return self.submod(x) + x


class PoseLifter(nn.Module):
    def __init__(self, num_joints, num_layers=2, num_features=4096):
        super(PoseLifter, self).__init__()

        self.num_joints = num_joints
        self.num_in = 2 * num_joints + 3
        self.num_out = 3 * (num_joints - 1) + 1
        self.num_layers = num_layers
        self.num_features = num_features

        mod = []
        mod.append(nn.Linear(self.num_in, num_features))
        for i in range(num_layers):
            mod.append(ResNetModule(num_features))
        mod.append(nn.Linear(num_features, self.num_out))

        # set weights
        nn.init.normal_(mod[0].weight, mean=0, std=0.001)
        nn.init.constant_(mod[0].bias, 0)
        nn.init.normal_(mod[-1].weight, mean=0, std=0.001)
        nn.init.constant_(mod[-1].bias, 0)

        self.mod = nn.ModuleList(mod)

    def forward(self, pose2d, cam_c):

        nb = pose2d.shape[0]
        pose2d = pose2d - torch.reshape(cam_c, (nb, 1, 2))

        mean2d = torch.mean(pose2d, 1, keepdim=True)
        dist = torch.sqrt(torch.sum((pose2d - mean2d) ** 2.0, 2, keepdim=True))
        std2d = torch.std(dist, 1, keepdim=True)
        pose2d = (pose2d - mean2d) / std2d
        mean2d = mean2d * 0.001
        std2d = std2d * 0.001

        x = torch.cat((pose2d.reshape(nb, -1), mean2d.reshape(nb, -1), std2d.reshape(nb, -1)), 1)

        x = self.mod[0](x)
        for i in range(self.num_layers):
            x = self.mod[i + 1](x)
        x = self.mod[-1](x)

        pose_local = x[:, 0:(self.num_out - 1)].view(-1, self.num_joints - 1, 3)
        depth_root = x[:, self.num_out - 1]

        return [pose_local, depth_root]


if __name__ == '__main__':
    model = PoseLifter(num_joints=17)

    pose2d = torch.randn(4, 17, 2)
    cam_c = torch.randn(4, 2)
    print(pose2d.shape)
    print(cam_c.shape)

    pose_local, depth_root = model(pose2d, cam_c)
    print(pose_local.shape)
    print(depth_root.shape)