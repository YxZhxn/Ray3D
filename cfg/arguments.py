

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--cfg', default='cfg_videopose', type=str, help='config file')
    parser.add_argument('--timestamp', default='', type=str, help='timestamp')
    parser.add_argument('--evaluate', default='', type=str, help='fill in the model name if in evaluation mode')
    parser.add_argument('--resume', default='', type=str, help='fill in the model name if need resume for training mode')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()

    return args
