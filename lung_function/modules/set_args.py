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


def get_args(jupyter=False):
    parser = argparse.ArgumentParser(description="SSc score prediction.")

    # Network
    parser.add_argument('--net', choices=('vgg11_3d', 'vit3', 'vgg16_3d', 'vgg19_3d', 'r3d_resnet', 'cnn3fc1', 'cnn4fc2',
                                          'cnn5fc2', 'cnn6fc2', 'cnn2fc1', 'cnn3fc2', 'r3d_18', 'slow_r50',
                                          'slowfast_r50', 'x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l', 'pointnet_reg','pointnet2_reg',
                                          'vgg11_3d', 'pointnext'),  # 'r2plus1d_18' out of memory
                        help='network name', type=str, default='x3d_m')
    
    # Point cloud network configuration
    parser.add_argument('--cfg', help='fpath of cfg',type=str, default='SSc_vessel/pointnext-s.yaml')  # ori = 40
    parser.add_argument('--fc2_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--fc1_nodes', help='the number of nodes of fc2 layer, original is 4096', type=int,
                        default=1024)
    parser.add_argument('--pointnet_fc_ls', help='a parameter list for fully connected layers. First number is the feature number after feature extraction', type=str, default="1024-512-256")
    parser.add_argument('--dp_fc1_flag', help='dropout for fc1',
                        type=boolean_string, default=True)
    parser.add_argument('--npoint_base', help='base of npoint',
                        type=int, default=512)  # ori = 512
    parser.add_argument('--radius_base', help='base of radius',
                        type=int, default=5)  # ori = 40
    parser.add_argument('--nsample_base', help='base of nsample',
                        type=int, default=32)  # ori = 64
    parser.add_argument('--width', help='width',
                        type=int, default=16)  # ori = 64
    parser.add_argument('--radius_scaling', help='base of radius',
                        type=float, default=2)  # ori = 2
    parser.add_argument('--sa_layers', help='sa_layers',
                        type=int, default=2)  # ori = 3

    # data
    # common data
    parser.add_argument('--batch_size', help='batch_size',
                        type=int, default=1)
    parser.add_argument('--ct_sp', help='space', type=str,
                        choices=('ori', '1.0', '1.5'), default='1.5')
    parser.add_argument('--kfold_seed', help='kfold_seed',
                        type=int, default=711)
    parser.add_argument('--test_pat', help='testing patients', choices=(
        'zhiwei77', 'random', 'random_as_ori'), type=str, default='random_as_ori')  # 
    parser.add_argument('--input_mode', help='what to input', 
        choices=('ct', 'ct_masked_by_torso', 'ct_left','ct_masked_by_lung','ct_masked_by_left_lung', 'ct_masked_by_right_lung', 
        'ct_right','ct_left_in_lung', 'ct_right_in_lung','ct_upper','ct_lower', 'ct_front', 'ct_back','ct_upper_in_lung',
        'ct_lower_in_lung', 'ct_front_in_lung', 'ct_back_in_lung', 'vessel', 'ct_masked_by_vessel', 'vessel_skeleton_pcd', 
        'ct_masked_by_vessel_dilated1', 'ct_masked_by_vessel_dilated2', 'ct_masked_by_vessel_dilated3', 'ct_masked_by_vessel_dilated4',
        'IntrA_cls'),
        type=str, default='ct')
    parser.add_argument('--target', help='target prediction', type=str,
                        default='FEV1-FVC-TLC_He')  # FVC-DLCO_SB-FEV1-TLC_He-Age-Height-Weight--DLCOc/pred-FEV1/pred-FVC/predNew-TLC/pred DLCOcPP-FEV1PP-FVCPP-TLCPP
    parser.add_argument(
        '--workers', help='number of workers for dataloader', type=int, default=6)

    # for gird image data
    parser.add_argument('--balanced_sampler', help='balanced_sampler', type=boolean_string, default='False')
    parser.add_argument('--crop_foreground', help='load lung mask, apply RandomCropForegroundd',
                        type=boolean_string, default='True')
    parser.add_argument(
        '--z_size', help='length of patch along z axil ', type=int, default=240)
    parser.add_argument(
        '--y_size', help='length of patch along y axil ', type=int, default=240)
    parser.add_argument(
        '--x_size', help='length of patch along x axil ', type=int, default=240)
    parser.add_argument('--pad_ratio', help='padding ratio',
                        type=float, default=1.5)

    # for point cloud data
    parser.add_argument('--shift_range', help='shift range', type=float, default=0)
    parser.add_argument('--PNB', help='points number for each image', type=int, default=56000)
    parser.add_argument('--FPS_input', help='Fartest point sample input', type=boolean_string, default='False')
    parser.add_argument('--repeated_sample', help='if apply repeated sampling to get PNB points?', type=boolean_string, default='False')
    parser.add_argument('--position_center_norm', help='if use the relative coordinates: center point is 0,0,0', type=boolean_string, default='True')


    # training parameters
    parser.add_argument('--mode', choices=('train', 'infer',
                        'continue_train'), help='mode', type=str, default='train')
    parser.add_argument('--pretrained_id', help='id used for inference, or continue_train',
                        type=str, default="0")  # SSc-852-853-854-855, 1504-1505-1510-1515
    # parser.add_argument('--reload_jobid', help='jobid used for inference, or continue_train', type=int, default=0)
    parser.add_argument('--pretrained_imgnet', help='if pretrained from imagenet',
                        type=boolean_string, default='True')
    parser.add_argument('--total_folds', choices=(4, 5),
                        help='4-fold training', type=int, default=4)
    parser.add_argument('--fold', choices=(1, 2, 3, 4),
                        help='1 to 4', type=int, default=1)
    parser.add_argument(
        '--valid_period', help='how many epochs between 2 validation', type=int, default=5)
    parser.add_argument('--loss', choices=('mse', 'mae', 'smooth_mae', 'mse+mae', 'msehigher', 'mse_regular'), help='mode', type=str,
                        default='mse')
    parser.add_argument('--mat_diff_loss_scale',
                        help='scale for another loss', type=float, default=0)
    parser.add_argument('--epochs', help='total epochs', type=int, default=500)
    parser.add_argument('--weight_decay', help='L2 regularization', type=float,
                        default=0.001)  # must be a float number !
    parser.add_argument('--lr', help='learning rate',
                        type=float, default=0.0001)
    parser.add_argument('--adamw', help='adamw optimizer',
                        type=boolean_string, default='False')
    parser.add_argument('--cosine_decay', help='cosine_decay',
                        type=boolean_string, default='False')
    # others
    parser.add_argument(
        '--outfile', help='output file when running by script instead of pycharm', type=str)
    parser.add_argument('--hostname', help='hostname of the server', type=str)
    parser.add_argument('--remark', help='comments on this experiment',
                        type=str, default='None')
    parser.add_argument('--jobid', help='slurm job_id', type=int, default=0)
    # For jupyter notebooks
    if jupyter:
        parser.add_argument(
            "--f", help="a dummy argument to fool ipython", default="0")

        args, unknown = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if args.x_size == 0 or args.y_size == 0:
        raise Exception("0 x_size or y_size: ")

    return args


if __name__ == "__main__":
    get_args()
