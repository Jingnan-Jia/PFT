# -*- coding: utf-8 -*-
# @Time    : 3/6/21 9:58 AM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# -*- coding: utf-8 -*-

import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser(description="SSc score prediction.")

    # Common args with set_args.py
    parser.add_argument('--mode', choices=('train', 'infer', 'continue_train'), help='mode', type=str, default='train')
    parser.add_argument('--pretrained_id', help='id used for inference, or continue_train', type=str, default='0') #SSc-852-853-854-855
    # parser.add_argument('--reload_jobid', help='jobid used for inference, or continue_train', type=int, default=0)
    parser.add_argument('--pretrained_imgnet', help='if pretrained from imagenet', type=boolean_string, default='False')
    parser.add_argument('--net', choices=('vgg11_3d','vit3', 'vgg16_3d','vgg19_3d', 'r3d_resnet', 'cnn3fc1', 'cnn4fc2',
                                          'cnn5fc2', 'cnn6fc2', 'cnn2fc1', 'cnn3fc2', 'r3d_18', 'slow_r50',
                                          'slowfast_r50', 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l', 'pointnet_reg', 'pointnet2_reg'),# 'r2plus1d_18' out of memory
                        help='network name', type=str, default='pointnet2_reg')
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
    parser.add_argument('--epochs', help='total epochs', type=int, default=1)
    parser.add_argument('--weight_decay', help='L2 regularization', type=float,
                        default=0.0001)  # must be a float number !
    parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
    parser.add_argument('--PNB', help='points number for each image', type=int, default=140000)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
    parser.add_argument('--ct_sp', help='space', type=str, choices=('ori', '1.0', '1.5'), default='ori')
    parser.add_argument('--kfold_seed', help='kfold_seed', type=int, default=711)
    parser.add_argument('--test_pat', help='testing patients', choices=('zhiwei77', 'random', 'random_as_ori'), type=str, default='random_as_ori')
    parser.add_argument('--input_mode', help='what to input', choices=('ct', 'vessel', 'ct_masked_by_vessel', 'vessel_skeleton_pcd', 'ct_masked_by_vessel_dilated1',
    'ct_masked_by_vessel_dilated2','ct_masked_by_vessel_dilated3','ct_masked_by_vessel_dilated4'), type=str, default='vessel_skeleton_pcd')

    parser.add_argument('--target', help='target prediction', type=str,
                        default='FVC-DLCO_SB-FEV1-TLC_He')  # FVC-DLCO_SB-FEV1-TLC_He-Age-Height-Weight--DLCOc/pred-FEV1/pred-FVC/predNew-TLC/pred

    parser.add_argument('--outfile', help='output file when running by script instead of pycharm', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment', type=str, default='None')
    parser.add_argument('--jobid', help='slurm job_id', type=int, default=0)
    parser.add_argument('--crop_foreground', help='crop_foreground', type=boolean_string, default='False')

    parser.add_argument('--z_size', help='length of patch along z axil ', type=int, default=240)
    parser.add_argument('--y_size', help='length of patch along y axil ', type=int, default=240)
    parser.add_argument('--x_size', help='length of patch along x axil ', type=int, default=240)
    parser.add_argument('--pad_ratio', help='padding ratio', type=float, default=1.5)

    args = parser.parse_args()

    if args.x_size == 0 or args.y_size == 0:
        raise Exception("0 x_size or y_size: ")

    return args


if __name__ == "__main__":
    get_args()