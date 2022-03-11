import visdom
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ExperimentPlotter(object):

    def __init__(self, data_config, model_config, train_config, plot_config, env='pose-detection-3d', port=8097):
        """

        :param data_config:
        :param model_config:
        :param train_config:
        :param plot_config:
        :param env:
        :param port:
        """
        self.visdom = visdom.Visdom(port=port, env=env)

        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.plot_config = plot_config

        self.plot_config_dict(self.data_config, 'Data Config: <br>')
        self.plot_config_dict(self.model_config, 'Model Config: <br>')
        self.plot_config_dict(self.train_config, 'Train Config: <br>')
        self.plot_config_dict(self.plot_config, 'Plot Config: <br>')

        self.plot_handles = self.initialize_plots()
        self.metrics = {
            'train': {'handle': 'loss', 'values': []},
            'test': {'handle': 'loss', 'values': []},

            'lr': {'handle': 'lr', 'values': []},

            'MPJPE': {'handle': 'metric', 'values': []},
            'P-MPJPE': {'handle': 'metric', 'values': []},
        }

    def plot_config_dict(self, config_dict, config_string):
        """

        :param config_dict:
        :param config_string:
        :return:
        """
        for config_item in config_dict:
            config_string += '<b>' + str(config_item) + '</b>: ' + str(config_dict[config_item]) + '<br>'
        config_win = self.visdom.text(config_string)
        return config_win

    def initialize_plots(self):
        """

        :return:
        """
        plot_handles = {}

        # initialize loss plot
        X = np.zeros((1, 2))
        Y = np.zeros((1, 2))
        opts = dict(
            legend=['train', 'test'],
            title='Losses',
            xlabel='Epoch',
            ylabel='Value',
            xtype='linear',
            ytype='linear',
            linecolor=np.array([
                [31, 119, 180], [255, 127, 14],
            ])
        )
        plot_handles['loss'] = self.visdom.line(X=X, Y=Y, opts=opts)

        # initialize learning rate plot
        X = [0]
        Y = [self.train_config['LEARNING_RATE']]
        X = np.array([X])
        Y = np.array([Y])
        opts = dict(
            legend=['lr'],
            title='Learning Rates',
            xlabel='Epoch',
            ylabel='Value',
            xtype='linear',
            ytype='linear',
            linecolor=np.array([[96, 96, 96]])
        )
        plot_handles['lr'] = self.visdom.line(X=X, Y=Y, opts=opts)

        # initialize metric plot
        X = np.zeros((1, 2))
        Y = np.zeros((1, 2))
        opts = dict(
            legend=['MPJPE', 'P-MPJPE'],
            title='Pose Metrics',
            xlabel='Epoch',
            ylabel='Value',
            xtype='liner',
            ytype='liner',
            linecolor=np.array([[17, 30, 108], [16, 52, 166]])
        )
        plot_handles['metric'] = self.visdom.line(X=X, Y=Y, opts=opts)

        return plot_handles

    def log_metric(self, metric, value, epoch):
        """

        :param metric:
        :param value:
        :param epoch:
        :return:
        """
        x = np.array([epoch])
        y = np.array([value])
        self.visdom.line(X=x, Y=y, win=self.plot_handles[self.metrics[metric]['handle']], name=metric, update='append')

    def show_plot(self, epoch, pose_2d, pose_3d_gt, pose_3d_pred, dataset, gt):
        """

        :param epoch:
        :param pose_2d:
        :param pose_3d_gt:
        :param pose_3d_pred:
        :param dataset:
        :param gt:
        :return:
        """
        if gt == 'gt':
            if dataset == 'h36m' or dataset == '3dhp':
                INDICES = [
                    (0, 1), (1, 2), (2, 3),
                    (0, 4), (4, 5), (5, 6),
                    (0, 7), (7, 8), (8, 9), (9, 10),
                    (8, 11), (11, 12), (12, 13),
                    (8, 14), (14, 15), (15, 16)
                ]
            if dataset == 'humaneva':
                INDICES = [
                    (0, 1), (1, 2), (2, 3),
                    (3, 4), (1, 5), (5, 6),
                    (6, 7), (0, 8), (8, 9), (9, 10),
                    (0, 11), (11, 12), (12, 13),
                    (1, 14)
                ]
        if gt == 'universal':
            INDICES = [
                (0, 1), (1, 2), (2, 3),
                (0, 4), (4, 5), (5, 6),
                (0, 7),
                (0, 8), (8, 9), (9, 10),
                (0, 11), (11, 12), (12, 13),
            ]
        COLORS = [
            (0 / 255., 215 / 255., 255 / 255.), (0 / 255., 255 / 255., 204 / 255.),
            (0 / 255., 134 / 255., 255 / 255.), (0 / 255., 255 / 255., 50 / 255.),
            (77 / 255., 255 / 255., 222 / 255.), (77 / 255., 196 / 255., 255 / 255.),
            (77 / 255., 135 / 255., 255 / 255.), (191 / 255., 255 / 255., 77 / 255.),
            (77 / 255., 255 / 255., 77 / 255.), (0 / 255., 127 / 255., 255 / 255.),
            (255 / 255., 127 / 255., 77 / 255.), (0 / 255., 77 / 255., 255 / 255.),
            (77 / 255., 255 / 255., 77 / 255.), (0 / 255., 127 / 255., 255 / 255.),
            (255 / 255., 127 / 255., 77 / 255.), (0 / 255., 77 / 255., 255 / 255.),
        ]

        def drawer_3d(ax, viz_3d):
            ax.scatter(viz_3d[0], viz_3d[1], viz_3d[2], 'b.')
            for ii, idx in enumerate(INDICES):
                ax.plot(
                    np.concatenate([viz_3d[0][idx[0]].reshape(1), viz_3d[0][idx[1]].reshape(1)]),
                    np.concatenate([viz_3d[1][idx[0]].reshape(1), viz_3d[1][idx[1]].reshape(1)]),
                    np.concatenate([viz_3d[2][idx[0]].reshape(1), viz_3d[2][idx[1]].reshape(1)]),
                    color=COLORS[ii][::-1]
                )
            # ax.set_xlabel('x axis')
            # ax.set_ylabel('y axis')
            # ax.set_zlabel('z axis')

        def drawer_2d(ax, viz_3d):
            ax.plot(viz_3d[0], viz_3d[1], 'b.')
            for ii, idx in enumerate(INDICES):
                ax.plot(
                    np.concatenate([viz_3d[0][idx[0]].reshape(1), viz_3d[0][idx[1]].reshape(1)]),
                    np.concatenate([viz_3d[1][idx[0]].reshape(1), viz_3d[1][idx[1]].reshape(1)]),
                    color=COLORS[ii][::-1]
                )
            # ax.set_xlabel('x axis')
            # ax.set_ylabel('y axis')

        assert pose_3d_gt.shape == pose_3d_pred.shape
        batch_size = pose_3d_gt.shape[0]

        batch_idx = np.random.randint(0, batch_size)

        f = plt.figure()

        ax = f.add_subplot(131, projection='3d')
        viz_3d = pose_3d_gt[batch_idx, 0].transpose()
        drawer_3d(ax, viz_3d)
        ax.title.set_text('3d_gt')

        ax = f.add_subplot(132, projection='3d')
        viz_3d = pose_3d_pred[batch_idx, 0].transpose()
        drawer_3d(ax, viz_3d)
        ax.title.set_text('3d_pred')

        ax = f.add_subplot(133)
        viz_3d = pose_2d[batch_idx, 0].transpose()
        plt.gca().invert_yaxis()
        drawer_2d(ax, viz_3d)
        ax.title.set_text('2d')

        plt.tight_layout()

        opts = dict(
            title='EPOCH.{ep:04d}'.format(ep=epoch),
        )

        self.visdom.matplot(plt, opts=opts, win='EPOCH.{ep:04d}'.format(ep=epoch))
        plt.close()

    def save_env(self):
        """

        :return:
        """
        self.visdom.save([self.visdom.env])