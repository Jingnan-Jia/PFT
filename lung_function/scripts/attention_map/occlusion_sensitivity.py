# -*- coding: utf-8 -*-
# @Time    : 3/28/21 2:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
from medutils.medutils import load_itk, save_itk
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import models
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import morphology
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info, retrive_run
from lung_function.modules.set_args import get_args
from lung_function.modules.path import PFTPath
from lung_function.modules.networks import get_net_3d
from lung_function.modules.loss import get_loss
from lung_function.modules.datasets import all_loaders
from lung_function.modules.trans import batch_bbox2_3D
from scipy.ndimage import gaussian_filter, median_filter
import scipy
from lung_function.scripts.run import Run
import time
import os
from monai.utils import set_determinism
import random
import numpy as np
from mlflow.tracking import MlflowClient
import statistics
import torch.nn as nn
import math
import copy
from tqdm import tqdm
from monai.transforms import ScaleIntensityRange
import pandas as pd
import torch
from queue import Queue
from medutils import medutils
from medutils.medutils import count_parameters
import time
import threading
import mlflow
from mlflow import log_metric, log_metrics, log_param, log_params
import sys
import argparse

sys.path.append("../../..")  #
print(os.getcwdb())

args = get_args()


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def masked_filter(method, img, sigma, mask):
    """
    method: 'gaussian' or 'median'
    img: image
    sigma: int
    mask: binary image with the same shape as img
    """
    # normalized convolution of image with mask
    if method == 'gaussian':
        filter = scipy.ndimage.filters.gaussian_filter(img * mask, sigma = sigma)
        weights = scipy.ndimage.filters.gaussian_filter(mask, sigma = sigma)
    elif method == 'median':
        filter = scipy.ndimage.filters.median_filter(img * mask, size = sigma)
        weights = scipy.ndimage.filters.median_filter(mask, size = sigma)
    else:
        raise Exception(f"filter method: {method}, but it should be 'gaussian' or 'median")
    filter /= weights
    # after normalized convolution, you can choose to delete any data outside the mask:
    filter *= mask
    return filter

def occlusion_map(ptch, data, net,  inlung=False, targets=None, occlusion_dir=None, save_occ_x=False, stride=None, occ_status='healthy', inputmode='ct'):
    """Save occlusion map to disk.

    Args:
        ptch: patch side lenth
        x: image to be predicted, shape [channel, w, h]
        y: predicted scores, shape [1, 3]
        net: network
        lung_mask: lung mask to ensure the occlusion occurs in lung area, shape [channel, w, h]
        occlusion_dir: directory to save occlusion maps

    Returns:
        None
    """

    if not os.path.isdir(occlusion_dir):
        os.makedirs(occlusion_dir)
    batch_x, y, lung_mask, ori, sp = data[inputmode][0], data['label'][0], data['lung_mask'][0], data['origin'][0], data['spacing'][0]
    input_mode = inputmode
    if input_mode == 'ct_masked_by_lung':
        a = copy.deepcopy(lung_mask)  # shape of batch_x and lung mask: [channel, z,y,x]
        a[a > 0] = 1
        batch_x += 1  # shift lowest value from -1 to 0
        batch_x = batch_x * a
        batch_x -= 1
    elif input_mode == 'ct_masked_by_left_lung':
        a = copy.deepcopy(lung_mask)
        a[a !=2] = 0
        batch_x += 1  # shift lowest value from -1 to 0
        batch_x = batch_x * a
        batch_x -= 1
    elif input_mode == 'ct_masked_by_right_lung':
        a = copy.deepcopy(lung_mask)
        a[a !=1] = 0
        batch_x += 1  # shift lowest value from -1 to 0
        batch_x = batch_x * a
        batch_x -= 1
    elif input_mode in ('ct_left', 'ct_right', 'ct_upper', 'ct_lower', 'ct_front', 'ct_back'):
        lung_mask = copy.deepcopy(lung_mask)
        lung_mask[lung_mask > 0] = 1
        if 'in_lung' in input_mode:  # only keep values in lung
            batch_x += 1  # shift lowest value from -1 to 0
            batch_x = batch_x * lung_mask  # masked by lung
            batch_x -= 1

        z_bottom, z_top, y_bottom, y_top, x_bottom, x_top = batch_bbox2_3D(lung_mask)
        z_mid, y_mid, x_mid = (z_bottom + z_top)//2, (y_bottom + y_top)//2, (x_bottom + x_top)//2
        # for idx in range(batch_x.shape[0]):
        if input_mode == 'ct_upper':
            batch_x[:, :z_mid[idx], :, :] = - 1  # remove bottom
        elif input_mode == 'ct_lower':
            batch_x[:, z_mid[idx]:, :, :] = - 1  # remove upper
        elif input_mode == 'ct_back':
            batch_x[:, :, y_mid[idx]:, :] = - 1  # remove front, keep back
        elif input_mode == 'ct_front':
            batch_x[:, :, :y_mid[idx], :] = - 1  # remove back, keep front
        elif input_mode == 'ct_left':
            batch_x[:, :, :, :x_mid[idx]] = - 1  # remove right
        else:  # args.input_mode == 'ct_front':
            batch_x[:, :, :, x_mid[idx]:] = - 1  # remove left
    else:
        pass
    x = batch_x[0]  # shape: x,y,z
    
    # lung_mask = morphology.binary_erosion(lung_mask.numpy(), np.ones((6, 6))).astype(int)
    if inlung:
        lung_mask = lung_mask.numpy()
        lung_mask[lung_mask > 0] = 1
        lung_mask[lung_mask <= 0] = 0
    x = x.to(device)  # shape [channel, w, h]
    net.to(device)
    x_ = x.unsqueeze(0).unsqueeze(0)  # add a channel and batch dims
    net.eval()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(): 
            out_ori = net(x_)
    l, w, h = x.shape
    # print(x.shape)  # [1, 512, 512] [channel, w, h]
    # print(out_ori.shape)  # [1, 3]

    # Three-pattern scores
    out_ori = out_ori.detach().cpu().numpy()
    nb_out = out_ori.shape[1]
    out_ori_all = out_ori.flatten()
    
    # out_ori = np.array([[1,2,3,4]])
    # nb_out = 1
    # out_ori_all = np.array([1,2,3,4])
    

    x_np = x.clone().detach().cpu().numpy()  # shape [l, w, h]
    map_all_ls = [np.zeros((l, w, h)) for i in range(nb_out)]
    map_all_w_ls = [np.zeros((l, w, h)) for i in range(nb_out)]

    pred_str = ""
    for t, p in zip(targets, out_ori_all):
        pred_str += f"{t}_{p:.2f}_"
    fpath_occ_x = f"{occlusion_dir}/no_occlusion_pred_{pred_str}.mha"
    save_itk(fpath_occ_x, x_np*1500, ori.tolist(), sp.tolist(), dtype='float')

    i, j, k = 0, 0, 0  # row index, column index
    img_lenth, img_width, img_height = 240, 240, 240
    lenth_steps, width_steps, height_steps = [(x - ptch)//stride + 2 for x in (img_lenth, img_width, img_height)]
    for a in tqdm(range(lenth_steps)):
        i = stride * a
        for b in tqdm(range(width_steps)):
            j = stride * b
            for c in tqdm(range(height_steps)):
                k = stride * c

                mask_ori = np.zeros((l, w, h))
                mask_ori[i: i + ptch, j: j + ptch, k: k + ptch] = 1
                if inlung:
                    erosed_lung = scipy.ndimage.binary_erosion(lung_mask[0], structure=np.ones((3,3,3))).astype(lung_mask[0].dtype)  # erosed_lung shape is [z, y, x]
                    mask_ori = mask_ori * erosed_lung  # exclude area outside lung, [z,y,x]
                # mask = cv2.blur(mask_ori, (5, 5)) # todo: 3d blur
                tmp = copy.deepcopy(x_np)
                shuffled_ls = tmp[i: i + ptch, j: j + ptch, k: k + ptch][mask_ori[i: i + ptch, j: j + ptch, k: k + ptch]>0]  # 1 dimensional
                if 'blur' in occ_status:
                    sigma = int(occ_status.split('_')[-1])
                    tmp_patch = copy.deepcopy(x_np[i: i + ptch, j: j + ptch, k: k + ptch])
                    mask_patch = mask_ori[i: i + ptch, j: j + ptch, k: k + ptch]
                    
                    if 'gaussian' in occ_status:
                        tmp_patch = masked_filter('gaussian', tmp_patch, sigma, mask_patch)
                        
                    elif 'median' in occ_status:
                        tmp_patch = masked_filter('median', tmp_patch, sigma, mask_patch)
                    
                    else:
                        raise Exception(f"wrong method of bluring: {occ_status}")
                    tmp[i: i + ptch, j: j + ptch, k: k + ptch] = tmp_patch

                elif 'shuffle' in occ_status:
                    
                    np.random.shuffle(shuffled_ls)
                    tmp[i: i + ptch, j: j + ptch, k: k + ptch][mask_ori[i: i + ptch, j: j + ptch, k: k + ptch]>0] = shuffled_ls
                elif 'erosion' in occ_status or 'dilation' in occ_status:
                    pass

                else:
                    raise Exception(f"wrong method of bluring: {occ_status}")
       

                new_x = torch.tensor(tmp).float()
                new_x = new_x.unsqueeze(0).unsqueeze(0)
                new_x = new_x.to(device)
                net.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():                
                        out = net(new_x)

                out_np = out.clone().detach().cpu().numpy()
                out_all = out_np.flatten()

                diff_all = out_all - out_ori_all

                if save_occ_x:
                    if len(shuffled_ls) > 0 and len(shuffled_ls) < ptch**3:  # do not save all steps
                        # if i%patch_size==0 and j%patch_size==0:  # do not save all steps
                        save_x_countor = True
                        tmp2 = copy.deepcopy(tmp)
                        if save_x_countor:  # show the contour of the image
                            edge = 1
                            contour_value = 1
                            tmp2[i:i + ptch, j:j + ptch,
                                 k:k + edge] = contour_value
                            tmp2[i:i + ptch, j:j + ptch, k +
                                 ptch:k + ptch - edge] = contour_value

                            tmp2[i:i + ptch, j:j + edge,
                                 k:k + ptch] = contour_value
                            tmp2[i:i + ptch, j + ptch:j + ptch -
                                 edge, k:k + ptch] = contour_value

                            tmp2[i:i + edge, j:j + ptch,
                                 k:k + ptch] = contour_value
                            tmp2[i + ptch:i + ptch - edge, j:j +
                                 ptch, k:k + ptch] = contour_value

                        pred_str = ""
                        for t, p in zip(targets, out_all):
                            pred_str += f"{t}_{p:.2f}_"
                        fpath_occ_x = f"{occlusion_dir}/{i}_{j}_{k}_pred_{pred_str}.mha"
                        save_itk(fpath_occ_x, tmp2*1500, ori.tolist(),
                                 sp.tolist(), dtype='float')

                for idx in range(len(map_all_ls)):
                    map_all_ls[idx][mask_ori > 0] += diff_all[idx]

                for idx in range(len(map_all_w_ls)):
                    map_all_w_ls[idx][mask_ori > 0] += 1

    for idx in range(len(map_all_w_ls)):
        map_all_w_ls[idx][map_all_w_ls[idx] == 0] = 1

    map_all_ls = [i/j for i, j in zip(map_all_ls, map_all_w_ls)]

    y_ls = list(y.numpy().flatten())
    pred_ls = list(out_ori_all.flatten())
    # print(y_ls,  '----')

    # per label
    for map, target, score, pred in zip(map_all_ls, targets, y_ls, pred_ls):
        fpath = f"{occlusion_dir}/{target}_label_{score: .2f}_pred_{pred: .2f}.mha"
        save_itk(fpath, map, ori.tolist(), sp.tolist(), dtype='float')


def get_pat_dir(img_fpath: str) -> str:
    dir_ls = img_fpath.split('/')
    for path in dir_ls:
        if 'Pat_' in path:  # "Pat_023"
            return path

def get_loader(mode, mypath, max_img_nb, args):
    label_all = pd.read_csv(mypath.save_label_fpath(mode))
    pred_all = pd.read_csv(mypath.save_pred_fpath(mode))
    mae_all = (label_all - pred_all).abs()
    mae_all['average'] = mae_all.mean(numeric_only=True, axis=1)
    label_all_sorted = label_all.loc[mae_all['average'].argsort()[:max_img_nb]]
    top_pats = label_all_sorted['pat_id'].to_list()

    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath, args, datasetmode='valid', top_pats=top_pats)
    dataloader = data_dt['valid']
    return dataloader



def batch_occlusion(args, mypath, myrun, patch_size: int, stride: int, max_img_nb: int, inlung, occ_status='healthy'):
    """Generate a batch of occlusion results

    Args:
        patch_size (int): _description_
        stride (int): _description_
        max_img_nb (int): _description_
        inlung (_type_): _description_
        occ_status (str, optional): _description_. Defaults to 'healthy'.
    """

    targets = [i.lstrip() for i in args.target.split('-')]  # FVC-DLCO_SB-FEV1-TLC_He
    
    valid_dataloader = get_loader(mode=mode, mypath=mypath, max_img_nb=max_img_nb, args=args)


    for data in tqdm(valid_dataloader):  # a image in a batch
        occlusion_map_dir = f"{mypath.id_dir}/{'valid_data_occlusion_maps_occ_by_masked_' + occ_status}/SSc_{data['pat_id'][0][0]}"
        occlusion_map(patch_size, data,
                        myrun.net,
                        inlung,
                        targets,
                        occlusion_map_dir,
                        save_occ_x=True,
                        stride=stride,
                        occ_status=occ_status,
                        inputmode=args.input_mode)


class Args:  # convert dict to class attribute
    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                if value == 'True':
                    value = True
                elif value == 'False':
                    value = False
                else:
                    try:
                        value = float(value)  # convert to float value if possible
                        try:
                            if int(value) == value:  # convert to int if possible
                                value = int(value)
                        except Exception:
                            pass
                    except Exception:
                        pass
                setattr(self, key, value)
                
def pre_setting(Ex_id):
    # retrive the run for the ex id
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    client = MlflowClient()
    run_ls = client.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.id LIKE '%{Ex_id}%'")
    run_ = run_ls[0]
    args_dt = run_.data.params  # extract the hyper parameters
    args = Args(args_dt)  # convert to object
    args.workers=1
    args.use_cuda = True
    mypath = PFTPath(Ex_id, check_id_dir=False, space=args.ct_sp)
    args.pretrained_id = Ex_id
    myrun = Run(args, dataloader_flag=False)
    
    return args, mypath, myrun

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)  # ori = 512
    args = parser.parse_args()
    
    Ex_id = args.id  # 2522 vgg 4-out, 2601 for vgg, 2657 for x3d_m FEV, 2666 for x3d_m of four outputs.
    max_img_nb = 3
    mode = 'valid'
    
    args, mypath, myrun = pre_setting(Ex_id)
     
    # for occ_status in ['shuffle', 'blur_median_5', 'blur_gaussian_5']:
    occ_status = 'shuffle'  # 'shuffle', 'constant' or 'healthy', or 'blur_median_*', or 'blur_gaussian_*', -1 is the minimum value
    patch_size = 16  # same for 3 dims
    stride = patch_size  # /2 or /4 to have high resolution heat map
    # grid_nb = 10

    # 2414->2415_fold1: ct_masked_by_torso
    # 2194->2195_fold1: ct
    # 2144->2145_fold1: ct_masked_by_lung
    # 2258->2259_fold1: vessel

    if args.input_mode in ['ct_masked_by_torso', 'ct']:
        INLUNG = False
    else:
        INLUNG = True

    batch_occlusion(args, mypath, myrun, patch_size, stride, max_img_nb=1,
                    inlung=INLUNG, occ_status=occ_status)
    print('finish all!')






