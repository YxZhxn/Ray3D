import visdom
import numpy as np


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
            'N-MPJPE': {'handle': 'metric', 'values': []},
            'MPJVE': {'handle': 'metric', 'values': []},
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
                [31, 119, 180], [255, 127, 14]
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
        X = np.zeros((1, 4))
        Y = np.zeros((1, 4))
        opts = dict(
            legend=['MPJPE', 'P-MPJPE', 'N-MPJPE', 'MPJVE'],
            title='Pose Metrics',
            xlabel='Epoch',
            ylabel='Value',
            xtype='liner',
            ytype='liner',
            linecolor=np.array([[17, 30, 108], [16, 52, 166], [124, 10, 2], [255, 36, 0]])
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

    def save_env(self):
        """

        :return:
        """
        self.visdom.save([self.visdom.env])
