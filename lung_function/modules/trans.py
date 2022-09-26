# -*- coding: utf-8 -*-
# @Time    : 7/5/21 4:01 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import os
import random
from typing import Dict, Optional, Union, Hashable, Sequence, Callable
from monai.utils import Method, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple

from medutils.medutils import load_itk, save_itk

import numpy as np
import pandas as pd
import torch
from monai.transforms import CropForeground, RandGaussianNoise, MapTransform, Transform, RandomizableTransform, ThreadUnsafe
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomAffine
from monai.transforms.utils import (
    allow_missing_keys_mode,
    generate_pos_neg_label_crop_centers,
    is_positive,
    map_binary_to_indices,
    weighted_patch_samples,
)
TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
# Note: all transforms here must inheritage Transform, Transform, or RandomTransform.

class SaveDatad(Transform):
    """Save the padded data, so that next time we can load the data directly, to save time.
    """
    def __init__(self, pad_truncated_dir):
        super().__init__()
        self.pad_truncated_dir = pad_truncated_dir

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data
        n = 7 - len(str(d['pat_id'][0]))
        if n>0:
            d['pat_id'] = [f"{'0'*n}{str(d['pat_id'][0])}"]
        fpath = f"{self.pad_truncated_dir}/SSc_patient_{d['pat_id'][0]}.nii.gz"
        save_itk(filename=fpath, scan=d['image'][0], origin=list(d['origin'].astype(np.float)), spacing=list(d['spacing'].astype(np.float)), dtype=float)
        fpath_lungmask = fpath.replace('.nii.gz', '_LungMask.nii.gz')
        save_itk(filename=fpath_lungmask, scan=d['lung_mask'][0], origin=list(d['origin'].astype(np.float)), spacing=list(d['spacing'].astype(np.float)))

        print(f"successfully save pad_truncated data to {fpath} and {fpath_lungmask}")
        return d


class LoadDatad(Transform):
    """Load data. The output image values range from -1500 to 1500.

        #. Load data from `data['fpath_key']`;
        #. truncate data image to [-1500, 1500];
        #. Get origin, spacing;
        #. Calculate relative slice number;
        #. Build a data dict.

    Examples:
        :func:`lung_function.modules.composed_trans.xformd_pos2score` and
        :func:`lung_function.modules.composed_trans.xformd_pos`
    """
    def __init__(self, target, crop_foreground=False):
        super().__init__()
        self.target = [i.lstrip() for i in target.split('-')]
        self.crop_foreground = crop_foreground
    def __call__(self, data: TransInOut) -> TransInOut:
        fpath = data['fpath']
        print(f"loading {fpath}")
        x, ori, sp = load_itk(fpath, require_ori_sp=True)  # shape order: z, y, x
        y = np.array([data[i] for i in self.target])
        # print(f"{fpath}, {y}")
        new_data = {'pat_id': np.array([int(fpath.split(".nii.gz")[0].split('_')[-1])]),  # extract the patient id as a string
                    'image': x.astype(np.float32),
                    'origin': ori.astype(np.float32),
                    'spacing': sp.astype(np.float32),
                    'label': y.astype(np.float32)}
        # new_data = {
        #             'image': x.astype(np.float32),
        #             'label': y.astype(np.float32)}
        if self.crop_foreground:
            lung_fpath = fpath.replace('.nii.gz', '_LungMask.nii.gz')
            lung_mask = load_itk(lung_fpath, require_ori_sp=False)  # shape order: z, y, x
            new_data['lung_mask'] = lung_mask.astype(np.float32)
        # print('load a image')
        # print("cliping ... ")
        # x[x < -1500] = -1500
        # x[x > 1500] = 1500
        # x = self.normalize0to1(x)
        # scale data to 0~1, it's convinent for future transform (add noise) during dataloader


        # data['ScanDate'] = str(data['ScanDate'])  # convert TimeStamp to string to avoid error during dataloader
        # data['PFT Date'] = str(data['PFT Date'])


        return new_data


def bbox2_3D(img):

    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    if rmax==0:
        rmax=img.shape[0]-1
    if cmax==0:
        cmax=img.shape[1]-1
    if zmax==0:
        zmax=img.shape[2]-1

    return rmin, rmax, cmin, cmax, zmin, zmax


class RandomCropForegroundd(MapTransform, RandomizableTransform):
    """
    Ensure that patch size is smaller than image size before this transform.
    """

    def __init__(
            self,
            keys,
            roi_size,
            source_key: str,
            allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.keys = keys
        self.z_size = roi_size[0]
        self.y_size = roi_size[1]
        self.x_size = roi_size[2]
        self.source_key = source_key

    def __call__(self, data) -> Dict:
        d = dict(data)
        zmin, zmax, ymin, ymax, xmin, xmax = bbox2_3D(d[self.source_key][0])  # remove channel dim
        z_lung = zmax-zmin
        y_lung = ymax-ymin
        x_lung = xmax-xmin

        z_res = self.z_size-z_lung
        y_res = self.y_size-y_lung
        x_res = self.x_size-x_lung

        rand_start = []
        for res, start in zip([z_res, y_res, x_res], [zmin, ymin, xmin]):
            if res > 0:
                shift = random.randint(max(start-res, 0), start)
            elif res ==0:
                shift = start
            else:
                shift = random.randint(start, start-res)
                print(f"patch size is smaller than lung size for pat {d['pat_id']}")
                # raise Exception(f"lung mask shape {d[self.source_key].shape} is smaller than patch size {self.z_size, self.y_size, self.x_size}")
            # elif res == 0:
            #     shift = 0
            # else:
            #     shift = random.randint(start, res)
            rand_start.append(shift)
        for key in self.keys:
            valid_start0 = min(rand_start[0], d[self.source_key][0].shape[0]-self.z_size)
            valid_start1 = min(rand_start[1], d[self.source_key][0].shape[1]-self.y_size)
            valid_start2 = min(rand_start[2], d[self.source_key][0].shape[2]-self.x_size)

            d[key] = d[key][:,
                     valid_start0: valid_start0 + self.z_size,
                     valid_start1: valid_start1 + self.y_size,
                     valid_start2: valid_start2 + self.x_size,]
        # del d[self.source_key]  # remove lung masks
        return d

class Clip:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, img):
        """
        Apply the transform to `img`.
        """

        img[img < self.min] = self.min
        img[img > self.max] = self.max
        return img


class Clipd(Transform):
    def __init__(self, min: int, max: int, key: str = 'image_key'):
        self.clip = Clip(min, max)
        self.key  = key

    def __call__(self, data):
        d = dict(data)
        d[self.key] = self.clip(d[self.key])
        return d
