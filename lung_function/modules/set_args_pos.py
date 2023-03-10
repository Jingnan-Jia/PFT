# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="SSc score prediction.")

    # Common args with set_args.py
    parser.add_argument('--mode', choices=('train', 'infer', 'continue_train'), help='mode', type=str, default='train')
    parser.add_argument('--eval_id', help='id used for inference, or continue_train', type=int, default=0)
    parser.add_argument('--net', choices=('vgg11_3d', 'r3d_resnet', 'cnn3fc1', 'cnn4fc2', 'cnn5fc2', 'cnn6fc2',
                                          'cnn2fc1', 'cnn3fc2'), help='network name', type=str, default='vgg11_3d')
    parser.add_argument('--fc2_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--fc1_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--total_folds', choices=(4, 5), help='4-fold training', type=int, default=4)
    parser.add_argument('--fold', choices=(1, 2, 3, 4), help='1 to 4', type=int, default=1)
    parser.add_argument('--valid_period', help='how many epochs between 2 validation', type=int, default=5)
    parser.add_argument('--workers', help='number of workers for dataloader', type=int, default=6)
    parser.add_argument('--loss', choices=('mse', 'mae', 'smooth_mae', 'mse+mae', 'msehigher'), help='mode', type=str,
                        default='mse')
    parser.add_argument('--pretrained', choices=(1, 0), help='pretrained or not', type=int, default=0)
    parser.add_argument('--steps', help='total epochs', type=int, default=50001)
    parser.add_argument('--weight_decay', help='L2 regularization', type=float,
                        default=0.0001)  # must be a float number !
    parser.add_argument('--batch_size', help='batch_size', type=int, default=5)
    parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment', type=str)

    parser.add_argument('--z_size', help='length of patch along z axil ', type=int, default=192)
    parser.add_argument('--y_size', help='length of patch along y axil ', type=int, default=256)
    parser.add_argument('--x_size', help='length of patch along x axil ', type=int, default=256)

    args = parser.parse_args()

    if args.x_size == 0 or args.y_size == 0:
        raise Exception("0 x_size or y_size: ")

    return args


if __name__ == "__main__":
    get_args()
