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
import itk
import os
import pdb
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

def marphology(input_fpath, method, output_fpath, radius=2):

    PixelType = itk.F
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]
    ReaderType = itk.ImageFileReader[ImageType]
    reader = ReaderType.New()
    reader.SetFileName(input_fpath)

    StructuringElementType = itk.FlatStructuringElement[Dimension]
    structuringElement = StructuringElementType.Ball(radius)
    if 'dilate' == method:
        GrayscaleFilterType = itk.GrayscaleDilateImageFilter[
            ImageType, ImageType, StructuringElementType
        ]
    elif 'erode' == method:
        GrayscaleFilterType = itk.GrayscaleErodeImageFilter[
            ImageType, ImageType, StructuringElementType
        ]
    else:
        raise Exception(f'wrong method {method}')
    grayscaleFilter = GrayscaleFilterType.New()
    grayscaleFilter.SetInput(reader.GetOutput())
    grayscaleFilter.SetKernel(structuringElement)

    WriterType = itk.ImageFileWriter[ImageType]
    writer = WriterType.New()
    writer.SetFileName(output_fpath)
    writer.SetInput(grayscaleFilter.GetOutput())

    writer.Update()
    pdb.set_trace()
    out = load_itk(output_fpath)

    out = out/1500
   
    return out 

def occlusion_map(data, net,  targets=None, occlusion_dir=None, save_occ_x=False, occ_status='healthy', inputmode='ct'):
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
    x = x.to(device)  # shape [channel, w, h]
    net.to(device)
    x_ = x.unsqueeze(0).unsqueeze(0)  # add a channel and batch dims
    out_ori = net(x_)


    # Three-pattern scores
    out_ori = out_ori.detach().cpu().numpy().flatten()

    x_np = x.clone().detach().cpu().numpy()  # shape [l, w, h]

    pred_str = ""
    for t, p in zip(targets, out_ori):
        pred_str += f"{t}_{p:.2f}_"
    fpath_occ_x = f"{occlusion_dir}/no_occlusion_pred_{pred_str}.mha"
    save_itk(fpath_occ_x, x_np*1500, ori.tolist(), sp.tolist(), dtype='float')

    x_np_dilasion = marphology(fpath_occ_x, 'dilate', output_fpath=f"{occlusion_dir}/global_dilated.mha")
    x_dilated = torch.tensor(x_np_dilasion).float()
    x_dilated = x_dilated.unsqueeze(0).unsqueeze(0)
    x_dilated = x_dilated.to(device)
    out_dilated = net(x_dilated)
    out_np_dilated = out_dilated.clone().detach().cpu().numpy().flatten()

    x_np_erosion = marphology(fpath_occ_x, 'erode', output_fpath=f"{occlusion_dir}/global_eroded.mha")
    x_eroded = torch.tensor(x_np_erosion).float()
    x_eroded = x_eroded.unsqueeze(0).unsqueeze(0)
    x_eroded = x_eroded.to(device)
    out_eroded = net(x_eroded)
    out_np_eroded = out_eroded.clone().detach().cpu().numpy().flatten()

    y_ls = list(y.numpy().flatten())
    pred_ls = list(out_ori)
    pred_dilated_ls = list(out_np_dilated)
    pred_eroded_ls = list(out_np_eroded)

    # per label
    for target, score, pred, pred_di, pred_er in zip(targets, y_ls, pred_ls, pred_dilated_ls, pred_eroded_ls):
        fpath = f"{occlusion_dir}/{target}_label_{score: .2f}_predOri_{pred: .2f}_predDilated_{pred_di: .2f}_predEroded_{pred_er: .2f}.mha"
        save_itk(fpath, x, ori.tolist(), sp.tolist(), dtype='float')


def get_pat_dir(img_fpath: str) -> str:
    dir_ls = img_fpath.split('/')
    for path in dir_ls:
        if 'Pat_' in path:  # "Pat_023"
            return path


def batch_occlusion(args, net_id: int, max_img_nb: int, occ_status='healthy'):
    """Generate a batch of occlusion results

    Args:
        net_id (int): _description_
        patch_size (int): _description_
        stride (int): _description_
        max_img_nb (int): _description_
        occ_status (str, optional): _description_. Defaults to 'healthy'.
    """

    targets = [i.lstrip() for i in args.target.split('-')
               ]  # FVC-DLCO_SB-FEV1-TLC_He
    net = get_net_3d(name=args.net, nb_cls=len(
        targets), image_size=args.x_size, pretrained=False)  # output FVC and FEV1

    mypath = PFTPath(net_id, space=args.ct_sp)  # get path
    mode = 'valid'
    label_all = pd.read_csv(mypath.save_label_fpath(mode))
    pred_all = pd.read_csv(mypath.save_pred_fpath(mode))
    mae_all = (label_all - pred_all).abs()
    mae_all['average'] = mae_all.mean(numeric_only=True, axis=1)
    label_all_sorted = label_all.loc[mae_all['average'].argsort()[:max_img_nb]]
    top_pats = label_all_sorted['pat_id'].to_list()

    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath,
                          args,datasetmode='valid', top_pats=top_pats)  # set nb to save time
    # only show visualization maps for valid dataset
    valid_dataloader = iter(data_dt[mode])

    ckpt = torch.load(mypath.model_fpath, map_location=device)
    if type(ckpt) is dict and 'model' in ckpt:
        model = ckpt['model']
    else:
        model = ckpt
    net.load_state_dict(model, strict=False)  # load trained weights
    net.eval()  # 8

    for data in tqdm(valid_dataloader):  # a image in a batch
        if int(data['pat_id']) in top_pats:  # only perform the most accurate patients
            occlusion_map_dir = f"{mypath.id_dir}/{'valid_data_global_' + occ_status}/SSc_{data['pat_id'][0][0]}"
            occlusion_map(data,
                          net,
                          targets,
                          occlusion_map_dir,
                          save_occ_x=True,
                          occ_status=occ_status,
                          inputmode=args.input_mode)


if __name__ == '__main__':
    # for occ_status in ['shuffle', 'blur_median_5', 'blur_gaussian_5']:
    occ_status = 'dilasion'  # 'erosion', 'constant' or 'healthy', or 'blur_median_*', or 'blur_gaussian_*', -1 is the minimum value

    # 2414->2415_fold1: ct_masked_by_torso
    # 2194->2195_fold1: ct
    # 2144->2145_fold1: ct_masked_by_lung
    # 2258->2259_fold1: vessel


    args = get_args()  # get argument
    args.batch_size = 1  # here batch size must be 1.
    args.net = 'vgg11_3d'
    args.target = 'FVC-DLCO_SB-FEV1-TLC_He'
    args.ct_sp = '1.5'

    id_input_dt = {
            2415: 'ct_masked_by_torso', 
            2195: 'ct', 
            2145: 'ct_masked_by_lung', 
            2259: 'vessel' }
    for id, im in id_input_dt.items():
        args.input_mode = im
        batch_occlusion(args, id, max_img_nb=1, occ_status=occ_status)
        print('---------------')
    print('finish all!')
