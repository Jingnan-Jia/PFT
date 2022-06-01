# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:31 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import glob
import random
from medutils.medutils import load_itk
from pathlib import Path
import numpy as np
from mlflow import log_metric, log_param, log_params

import torch
# import streamlit as st
from tqdm import tqdm
from monai.data import DataLoader
import random
from monai.transforms import RandSpatialCropd, RandGaussianNoised, CastToTyped, ToTensord, \
    CenterSpatialCropd, AddChanneld,ScaleIntensityRanged, SpatialPadd
import pandas as pd
from torch.utils.data import Dataset
import monai
from sklearn.model_selection import KFold
from lung_function.modules.trans import LoadDatad, RandomCropForegroundd

def xformd(mode, z_size: int = 192, y_size: int = 256, x_size: int = 256, target='FVC', crop_foreground=False):
    pad_ratio = 1.5
    log_param('pad_ratio', pad_ratio)
    post_pad_size = [int(i * pad_ratio) for i in [z_size, y_size, x_size]]

    keys = ('image', 'lung_mask')
    xforms = [LoadDatad(target=target, crop_foreground=crop_foreground), AddChanneld(keys=keys)]
    xforms.extend([SpatialPadd(keys=keys, spatial_size=post_pad_size, mode='minimum'),
                   ScaleIntensityRanged(keys=keys, a_min=-1500, a_max=1500, b_min=-1, b_max=1, clip=True)])
    if mode == 'train':
        if crop_foreground:
            xforms.extend([RandomCropForegroundd(keys=keys, roi_size=[z_size, y_size, x_size], source_key='lung_mask')])
        else:
            xforms.extend([RandSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size], random_center=True, random_size=False)])
        # xforms.extend([RandGaussianNoised(keys=keys, prob=0.5, mean=0, std=0.01)])
    else:
            xforms.extend([CenterSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size])])

    xforms.extend([CastToTyped(keys = keys, dtype=np.float32),
                   ToTensord(keys = keys)])
    transform = monai.transforms.Compose(xforms)

    return transform

def clean_data(pft_df, data_dir):
    pft_df.drop(pft_df[np.isnan(pft_df.DLCO_SB)].index, inplace=True)
    pft_df.drop(pft_df[pft_df.DLCO_SB == 0].index, inplace=True)
    pft_df.drop(pft_df[np.isnan(pft_df['FEV 1'])].index, inplace=True)
    pft_df.drop(pft_df[pft_df['FEV 1'] == 0].index, inplace=True)
    pft_df.drop(pft_df[np.isnan(pft_df.DateDF_abs)].index, inplace=True)
    pft_df.drop(pft_df[pft_df.DateDF_abs > 10].index, inplace=True)

    scans = glob.glob(data_dir + "/SSc*[!LungMask].nii.gz")  # exclude lung mask files
    availabel_id_set = set([Path(id).stem[:-4] for id in scans])  # use stem and :-4 to remove .nii.gz
    pft_df.drop(pft_df.loc[~pft_df['subjectID'].isin(availabel_id_set)].index, inplace=True)


    # pft_df = pft_df.drop(pft_df[pft_df['subjectID'] not in availabel_id_set].index)

    assert len(scans)==len(pft_df)

    return pft_df

def all_loaders(data_dir, label_fpath, args):
    label_excel = pd.read_excel(label_fpath, engine='openpyxl')
    label_excel = clean_data(label_excel, data_dir)
    # 3 labels for one level
    data = np.array(label_excel.to_dict('records'))  # nparray is easy for kfold split
    for d in data:
        d['fpath'] = data_dir + '/' + d['subjectID'] + '.nii.gz'
    # sub_id = pd.DataFrame(label_excel, columns=['subjectID']).values
    # sub_id = [i[0][-7:] for i in sub_id]
    #
    # # dt_excel = pd.read_excel(id_dt_file, engine='openpyxl')
    # # study_id_ls = pd.DataFrame(dt_excel, columns=['StudyNo']).values
    # # sub_id_ls = pd.DataFrame(dt_excel, columns=['Patno']).values
    # # sub_study_dt = {key: value for key in sub_id_ls for value in study_id_ls}
    # mypath = 3
    # study_id = [sub_study_dt[i] for i in sub_id]
    # study_id_fpath = [id_fpath(i) for i in study_id]
    # data = [{'id':id, 'DLCO': , 'fpath': mypath.data_dir + '/' + id} for id in sub_id]
    random.shuffle(data)
    ts_nb = int(0.2 * len(data))
    tr_vd_data, ts_data = data[:-ts_nb], data[-ts_nb:]

    kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=args.kfold_seed)  # for future reproduction
    kf_list = list(kf.split(tr_vd_data))
    tr_pt_idx, vd_pt_idx = kf_list[args.fold - 1]
    tr_data = tr_vd_data[tr_pt_idx]
    vd_data = tr_vd_data[vd_pt_idx]
    tr_data, vd_data, ts_data = tr_data[:5], vd_data[:5], ts_data[:5]
    # trxformd = xformd('train')
    # vdxformd = xformd('valid')
    # tsxformd = xformd('test')

    tr_dataset = monai.data.CacheDataset(data=tr_data, transform=xformd('train', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, target=args.target, crop_foreground=args.crop_foreground), num_workers=args.workers, cache_rate=1)
    vd_dataset = monai.data.CacheDataset(data=vd_data, transform=xformd('valid', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, target=args.target, crop_foreground=args.crop_foreground), num_workers=args.workers, cache_rate=1)
    ts_dataset = monai.data.CacheDataset(data=ts_data, transform=xformd('test', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, target=args.target, crop_foreground=args.crop_foreground), num_workers=args.workers, cache_rate=1)
    # training dataset without any data augmentation to simulate the valid transform to see if center crop helps
    tr_dataset_no_aug = monai.data.CacheDataset(data=tr_data, transform=xformd('valid', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, target=args.target, crop_foreground=args.crop_foreground), num_workers=args.workers, cache_rate=1)

    train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  persistent_workers=True)
    test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                 persistent_workers=True)

    train_dataloader_no_aug = DataLoader(tr_dataset_no_aug, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                  persistent_workers=True)

    data_dt = {'train': train_dataloader,
               'valid': valid_dataloader,
               'test': test_dataloader,
               'trainnoaug': train_dataloader_no_aug}
    return data_dt

