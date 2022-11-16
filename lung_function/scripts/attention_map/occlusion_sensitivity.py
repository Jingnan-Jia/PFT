# -*- coding: utf-8 -*-
# @Time    : 3/28/21 2:18 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import sys
sys.path.append("../../..")  #
from mlflow import log_metric, log_metrics, log_param, log_params
import mlflow
import threading
import time
from medutils.medutils import count_parameters
from medutils import medutils
from queue import Queue
import torch

from monai.transforms import ScaleIntensityRange
from tqdm import tqdm
import copy
import math
import time
import torch.nn as nn
import copy
import statistics
from mlflow.tracking import MlflowClient
import numpy as np
import random
from monai.utils import set_determinism
import os
print(os.getcwdb())

from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info, retrive_run
from lung_function.modules.compute_metrics import icc, metrics
args = get_args()


from scipy.ndimage import morphology
import matplotlib.pyplot as plt

import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
import json
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath

from torch.utils.data import Dataset, DataLoader
from medutils.medutils import load_itk, save_itk

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

grad_block = []
fmap_block = []
# 图片预处理
def img_preprocess(img):
    img_out = (img - np.mean(img)) / np.std(img)  # normalize
    img_out = img_out[None]
    img_out = torch.as_tensor(img_out)

    return img_out


# 定义获取梯度的函数
def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)


def apply_custom_colormap(image_gray, cmap=plt.get_cmap('seismic')):

    assert image_gray.dtype == np.uint8, 'must be np.uint8 image'
    if image_gray.ndim == 3: image_gray = image_gray.squeeze(-1)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:,0:3]    # color range RGBA => RGB
    color_range = (color_range*255.0).astype(np.uint8)         # [0,1] => [0,255]
    color_range = np.squeeze(np.dstack([color_range[:,2], color_range[:,1], color_range[:,0]]), 0)  # RGB => BGR

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image_gray, color_range[:,i]) for i in range(3)]
    return np.dstack(channels)


def cam_show_img(img, feature_map, grads, out_dir, idx):
    img = img.cpu().numpy()
    _, __, H, W = img.shape  # (1,1,512,512)
    img = np.resize(img, (H, W, 1))
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 4
    grads = grads.reshape([grads.shape[0], -1])  # 5
    weights = np.mean(grads, axis=1)  # 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]  # 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img_jpg = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255

    cam_img = 0.3 * heatmap + 0.7 * img_jpg
    heatmap_img = os.path.join(out_dir, str(idx) + "_cam.jpg")
    path_cam_img = os.path.join(out_dir, str(idx) + "_img_cam.jpg")
    path_img = os.path.join(out_dir, str(idx) + "_img.jpg")

    cv2.imwrite(heatmap_img, heatmap)
    cv2.imwrite(path_cam_img, cam_img)
    cv2.imwrite(path_img, img_jpg)


def grad_cam(x, y, net, nb_img):
    # fmap_block = []
    # grad_block = []
    x = x[None]
    y = y[None]
    print(y.cpu().numpy())

    # 注册hook, vgg has features and classifiers
    net.features[3].register_forward_hook(farward_hook)
    net.features[3].register_backward_hook(backward_hook)

    # forward
    output = net(x)

    # backward
    net.zero_grad()
    loss_fun = torch.nn.MSELoss()
    class_loss = loss_fun(output, y)
    class_loss.backward()

    # 生成cam
    print(len(grad_block))
    grads_val = grad_block[0].cpu().data.cpu().numpy().squeeze()
    fmap = fmap_block[0].cpu().data.cpu().numpy().squeeze()
    cam_show_img(x, fmap, grads_val, mypath.id_dir, nb_img)


def generate_candidate(fpath: str, image_size: int = 512):
    """

    Args:
        fpath: full path of seed patch

    Returns:
        Filled image, shape: [512, 512]
    """
    ori_image_fpath = fpath.split('.mha')[0] + '_ori.mha'
    egg = load_itk(fpath)
    # ori = load_itk(ori_image_fpath)
    normalize0to1 = ScaleIntensityRange(a_min=-1500.0, a_max=1500.0, b_min=0.0, b_max=1.0, clip=True)
    egg = normalize0to1(egg)
    # egg[egg > 1500] = 1500
    # egg[egg < -1500] = -1500
    # ori[ori > 1500] = 1500
    # ori[ori < -1500] = -1500
    # # normalize the egg using the original image information
    # egg = (egg - np.min(ori)) / (np.max(ori) - np.min(ori))  # rescale to [0, 1]

    minnorv = np.vstack((np.flip(egg), np.flip(egg, 0)))
    minnorh = np.hstack((minnorv, np.flip(minnorv, 1)))

    cell_size = minnorh.shape
    nb_row, nb_col = image_size // cell_size[0] * 2, image_size // cell_size[
        1] * 2  # big mask for crop
    temp = np.hstack(([minnorh] * nb_col))
    temp = np.vstack(([temp] * nb_row))
    temp = temp[:image_size, :image_size]
    return temp


def savefig(save_flag: bool, img: np.ndarray, image_name: str, dir: str = "image_samples") -> None:
    """Save figure.

    Args:
        save_flag: Save or not.
        img: Image numpy array.
        image_name: image name.
        dir: directory

    Returns:
        None. Image will be saved to disk.

    Examples:
        :func:`ssc_scoring.mymodules.data_synthesis.SysthesisNewSampled`

    """

    fpath = os.path.join(dir, image_name)
    # print(f'image save path: {fpath}')

    directory = os.path.dirname(fpath)
    if not os.path.isdir(directory):
        os.makedirs(directory)

    if save_flag:
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        fig.savefig(fpath, bbox_inches='tight')
        plt.close()


def occlusion_map(ptch, data, net,  inlung=False, targets=None, occlusion_dir=None, save_occ_x=False, stride=None,occ_status='healthy'):
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
    x, y, lung_mask, ori, sp = data['image'][0], data['label'][0], data['lung_mask'][0], data['origin'][0], data['spacing'][0]

    # lung_mask = morphology.binary_erosion(lung_mask.numpy(), np.ones((6, 6))).astype(int)
    if inlung:
        lung_mask = lung_mask.numpy()
        lung_mask[lung_mask > 0] = 1
        lung_mask[lung_mask <= 0] = 0
    x = x.to(device)  # shape [channel, w, h]
    net.to(device)
    x_ = x.unsqueeze(0)  # add a batch channel
    out_ori = net(x_)
    _,l, w, h = x.shape
    # print(x.shape)  # [1, 512, 512] [channel, w, h]
    # print(out_ori.shape)  # [1, 3]

    # Three-pattern scores
    out_ori = out_ori.detach().cpu().numpy()
    nb_out = out_ori.shape[1]
    out_ori_all = out_ori.flatten()

    x_np = x.clone().detach().cpu().numpy()  # shape [channel, l, w, h]
    x_np = x_np[0]  # shape [l, w, h]
    map_all_ls = [np.zeros((l, w, h)) for i in range(nb_out)]
    map_all_w_ls = [np.zeros((l, w, h)) for i in range(nb_out)]

    if occ_status=='healthy':
        occ_seed = "/home/jjia/data/ssc_scoring/ssc_scoring/dataset/special_samples/healthy/healthy.mha"
        occ_patch = generate_candidate(occ_seed)  # the healthy image is filled by healthy patches
    elif 'constant' in occ_status:
        cons_value = int(occ_status.split('_')[-1])  # 'constant_-1'
        # pass # todo: add constant status
        occ_patch = np.ones((l, w, h)) * cons_value

    pred_str = ""
    for t, p in zip(targets, out_ori_all):
        pred_str += f"{t}_{p:.2f}_"
    fpath_occ_x = f"{occlusion_dir}/no_occlusion_pred_{pred_str}.mha"
    save_itk(fpath_occ_x, x_np*1500, ori.tolist(), sp.tolist(), dtype='float')


    i, j, k = 0, 0, 0  # row index, column index
    img_lenth, img_width, img_height = 240, 240, 240
    lenth_steps, width_steps, height_steps=[(x - ptch)//stride + 2 for x in (img_lenth, img_width, img_height)]
    for a in tqdm(range(lenth_steps)):
        i = stride * a 
        for b in tqdm(range(width_steps)):
            j = stride * b 
            for c in tqdm(range(height_steps)):
                k = stride * c 

                mask_ori = np.zeros((l, w, h))
                mask_ori[i : i + ptch, j: j + ptch, k: k + ptch] = 1
                if inlung:
                    mask_ori = mask_ori * lung_mask # exclude area outside lung
                # mask = cv2.blur(mask_ori, (5, 5)) # todo: 3d blur
                mask = mask_ori
                tmp = x_np * (1-mask) + occ_patch * mask

                new_x = torch.tensor(tmp).float()
                new_x = new_x.unsqueeze(0).unsqueeze(0)
                new_x = new_x.to(device)
                out = net(new_x)

                out_np = out.clone().detach().cpu().numpy()
                out_all = out_np.flatten()

                diff_all = out_all - out_ori_all

                if save_occ_x:
                    if i==128 and j==128 and k==128: # do not save all steps
                    # if i%patch_size==0 and j%patch_size==0:  # do not save all steps
                        save_x_countor = False
                        tmp2 = copy.deepcopy(tmp)
                        if save_x_countor:  # show the contour of the image 
                            edge = 5
                            contour_value = 1
                            tmp2[i:i + ptch, j:j + ptch, k:k + edge] = contour_value
                            tmp2[i:i + ptch, j:j + ptch, k+ ptch:k + ptch - edge] = contour_value

                            tmp2[i:i + ptch, j:j + edge, k:k + ptch] = contour_value
                            tmp2[i:i + ptch, j+ ptch:j + ptch - edge, k:k + ptch] = contour_value

                            tmp2[i:i + edge, j:j + ptch, k:k + ptch] = contour_value
                            tmp2[i+ ptch:i + ptch - edge, j:j + ptch, k:k + ptch] = contour_value

                        pred_str = ""
                        for t, p in zip(targets, out_all):
                            pred_str += f"{t}_{p:.2f}_"
                        fpath_occ_x = f"{occlusion_dir}/{i}_{j}_pred_{pred_str}.mha"
                        save_itk(fpath_occ_x, tmp2*1500, ori.tolist(), sp.tolist(), dtype='float')

                for idx in range(len(map_all_ls)):
                    map_all_ls[idx][mask_ori > 0] += diff_all[idx]

                for idx in range(len(map_all_w_ls)):
                    map_all_w_ls[idx][mask_ori > 0] += 1

    for idx in range(len(map_all_w_ls)):
        map_all_w_ls[idx][map_all_w_ls[idx] == 0] = 1

    map_all_ls = [i/j for i, j in zip(map_all_ls, map_all_w_ls)]

    y_ls = list(y.numpy().flatten())
    pred_ls = list(out_np.flatten())
    # print(y_ls,  '----')

    for map, target, score, pred in zip(map_all_ls, targets, y_ls, pred_ls):  # per label
        fpath = f"{occlusion_dir}/{target}_label_{score: .2f}_pred_{pred: .2f}.mha"
        save_itk(fpath, map*1000, ori.tolist(), sp.tolist(), dtype='float')

def get_pat_dir(img_fpath: str) -> str:
    dir_ls = img_fpath.split('/')
    for path in dir_ls:
        if 'Pat_' in path:  # "Pat_023"
            return path



def batch_occlusion(net_id: int, patch_size: int, stride: int, max_img_nb: int, inlung, occ_status='healthy'):

    args = get_args()  # get argument
    targets = [i.lstrip() for i in args.target.split('-')]
    net = get_net_3d(name=args.net, nb_cls=len(targets), image_size=args.x_size, pretrained=False)  # output FVC and FEV1

    mypath = PFTPath(net_id)  # get path
    data_dt = all_loaders(mypath.data_dir, mypath.label_fpath, args, nb=3)
    valid_dataloader = iter(data_dt['valid'])  # only show visualization maps for valid dataset

    ckpt = torch.load(mypath.model_fpath, map_location=device)
    if type(ckpt) is dict and 'model' in ckpt:
        model = ckpt['model']
    else:
        model = ckpt
    net.load_state_dict(model)  # load trained weights
    net.eval()  # 8

    nb_img= 0
    for data in tqdm(valid_dataloader):  # a image in a batch
        nb_img += 1
        if nb_img > max_img_nb:
            break

        occlusion_map_dir = f"{mypath.id_dir}/{'valid_data_occlusion_maps_occ_by_' + occ_status}/SSc_{data['pat_id'][0]}"
        occlusion_map(patch_size, data,
                      net,
                      inlung,
                      targets,
                      occlusion_map_dir,
                      save_occ_x=True,
                      stride=stride,
                      occ_status=occ_status)





if __name__ == '__main__':
    occ_status = 'constant_-1' # 'constant' or 'healthy', -1 is the minimum value
    id = 905
    INLUNG = False
    patch_size = 64  # same for 3 dims
    stride = patch_size//4  # /2 or /4 to have high resolution heat map
    # grid_nb = 10
    batch_occlusion(id, patch_size, stride, max_img_nb=3, inlung=INLUNG, occ_status=occ_status)
    print('finish!')
