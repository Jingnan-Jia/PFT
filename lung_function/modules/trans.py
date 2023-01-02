# -*- coding: utf-8 -*-
# @Time    : 7/5/21 4:01 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import random
from typing import Dict, Optional, Union, Hashable

from medutils.medutils import load_itk, save_itk
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage.morphology import binary_dilation
from monai.transforms import MapTransform, Transform, RandomizableTransform
TransInOut = Dict[Hashable, Optional[Union[np.ndarray, torch.Tensor, str, int]]]
# Note: all transforms here must inheritage Transform, Transform, or RandomTransform.


class RemoveTextd(MapTransform):
    """
    Remove the text to avoid the Error: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U80
    """
    def __init__(self, keys):
        super().__init__(keys, allow_missing_keys=True)

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data
        for key in self.keys:
            del d[key] 
        return d
 

class SaveDatad(MapTransform):
    """Save the padded data, so that next time we can load the data directly, to save time.
    """
    def __init__(self, keys, pad_truncated_dir, crop_foreground=True, inputmode=None):
        super().__init__(keys, allow_missing_keys=True)
        self.pad_truncated_dir = pad_truncated_dir
        self.crop_foreground=crop_foreground
        self.inputmode = inputmode

    def __call__(self, data: TransInOut) -> TransInOut:
        d = data
        for key in self.keys:
            fpath = f"{self.pad_truncated_dir}/{Path(str(d['fpath'][0])).name}"
            save_itk(filename=fpath, scan=d[key][0], origin=list(d['origin'].astype(np.float)), spacing=list(d['spacing'].astype(np.float)), dtype=float)
            print(f"successfully save pad_truncated data to {fpath}")
            if self.crop_foreground:
                if key=='vessel':
                    fpath_lungmask = fpath.replace('_GcVessel.nii.gz', '_LungMask.nii.gz')
                elif 'ct' in key:
                    fpath_lungmask = fpath.replace('.nii.gz', '_LungMask.nii.gz')
                else:
                    raise Exception(f"please input proper key")
                save_itk(filename=fpath_lungmask, scan=d['lung_mask'][0], origin=list(d['origin'].astype(np.float)), spacing=list(d['spacing'].astype(np.float)))

                print(f"successfully save pad_truncated data to {fpath_lungmask}")
        return d


class ShiftCoordinated(MapTransform):
    """
    Shift the coordinate to the center of the image.
    """
    def __init__(self, keys, position_center_norm):

        super().__init__(keys, allow_missing_keys=True)
        self.position_center_norm = position_center_norm

    def __call__(self, data: TransInOut) -> TransInOut:
        for key in self.keys:  
            if self.position_center_norm:  # shuffle all points
                data[key] = data[key]

        return data


class SampleShuffled(MapTransform, RandomizableTransform):
    """
    Randomly shuffle the location data.
    """
    def __init__(self, keys, PNB, total_shuffle=True, sub_shuffle=True):

        super().__init__(keys, allow_missing_keys=True)
        assert PNB > 0
        self.PNB = PNB
        self.total_shuffle = total_shuffle
        self.sub_shuffle = sub_shuffle

    def __call__(self, data: TransInOut) -> TransInOut:
        print("running random shu")
        for key in self.keys:  
            if self.total_shuffle:  # shuffle all points
                np.random.shuffle(data[key])    # shuffle data inplace

            data[key] = data[key][:self.PNB]  # sample data

            if self.sub_shuffle:  # shuffle the sub data
                np.random.shuffle(data[key])    # shuffle data inplace
           
        return data

class LoadPointCloud(MapTransform):
    def __init__(self, keys, target, position_center_norm):
        super().__init__(keys, allow_missing_keys=True)
        self.target = [i.lstrip() for i in target.split('-')]
        self.position_center_norm = position_center_norm


    def __call__(self, data: TransInOut) -> TransInOut:
        fpath = data['fpath']
        # print(f"loading {fpath}")
        xyzr = pd.read_pickle(fpath)

        xyz_mm = xyzr['data'][:,:3] * xyzr['spacing']  # convert voxel location to physical mm
        if self.position_center_norm:
            xyz_mm -= xyz_mm.mean(axis=0)
        xyzr_mm = np.concatenate((xyz_mm, xyzr['data'][:,-1].reshape(-1,1)), axis=1)
        y = np.array([data[i] for i in self.target])
        file_id = fpath.split(".nii.gz")[0].split('SSc_patient_')[-1].split('_')[0]
        new_data = {'pat_id': np.array([int(file_id)]),  # Note: save it as a array. extract the patient id as a int, otherwise, error occured: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U21
                    self.keys[0]: xyzr_mm.astype(np.float32),
                    'origin': xyzr['origin'],
                    'spacing': xyzr['spacing'],
                    'label': y.astype(np.float32),
                    'fpath': np.array([fpath])}

        return new_data


def convertfpath(ori_path):
    if 'GcVessel' in ori_path:
        return ori_path.replace('_GcVessel.nii.gz', '_LungMask.nii.gz')
    else:
        return ori_path.replace('.nii.gz', '_LungMask.nii.gz')


class LoadDatad(MapTransform):
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
    def __init__(self, keys, target, crop_foreground=False, inputmode='image'):
        super().__init__(keys, allow_missing_keys=True)
        self.target = [i.lstrip() for i in target.split('-')]
        self.crop_foreground = crop_foreground
        self.inputmode = inputmode

    def __call__(self, data: TransInOut) -> TransInOut:
        fpath = data['fpath']
        print(f"loading {fpath}")
        x, ori, sp = load_itk(fpath, require_ori_sp=True)  # shape order: z, y, x
        if 'ct_masked_by_vessel' in self.inputmode:
            vessel_mask, vessel_ori, vessel_sp = load_itk(fpath.replace('.nii.gz', '_GcVessel.nii.gz'), require_ori_sp=True) 
            if self.inputmode[-1] in ['1', '2', '3', '4', '5', '6']:
                dilation_factor = int(self.inputmode[-1])
                for i in range(dilation_factor):  # dilate several times
                    vessel_mask = binary_dilation(vessel_mask)
            # assert np.linalg.norm(ori - vessel_ori) < 1e-6
            assert np.linalg.norm(sp - vessel_sp) < 1e-6
            # if '6339687' in fpath:
            #     print(fpath, 'vessel mask multplication')
            # try:
            x = x * vessel_mask
            # except ValueError:
            #     print('yes')
            x[vessel_mask<=0] = -1500  # set non-vessel as -1500
        if 'vessel' in self.inputmode:  # vessel needs to be masked by lung erosion to remove noises at the edges
            lung_fpath = convertfpath(fpath)
            lung_mask = load_itk(lung_fpath, require_ori_sp=False)  # shape order: z, y, x
            lung_mask[lung_mask>0] = 1  # lung mask may include 1 for left lung and 2 for right lung
            x += 1500  # shift all values
            x = x * lung_mask
            x -= 1500

            # save_itk(fpath.replace('.nii.gz', '_GcVessel_dilated.nii.gz'), x, ori, sp)

        y = np.array([data[i] for i in self.target])
        # print(f"{fpath}, {y}")
        file_id = fpath.split(".nii.gz")[0].split('SSc_patient_')[-1].split('_')[0]
        new_data = {'pat_id': np.array([int(file_id)]),  # Note: save it as a array. extract the patient id as a int, otherwise, error occured: TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <U21
                    self.keys[0]: x.astype(np.float32),
                    'origin': ori.astype(np.float32),
                    'spacing': sp.astype(np.float32),
                    'label': y.astype(np.float32),
                    'fpath': np.array([fpath])}
        # new_data = {s
        #             'image': x.astype(np.float32),
        #             'label': y.astype(np.float32)}
        if self.crop_foreground:

            lung_fpath = convertfpath(fpath)
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
