# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:31 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import sys
sys.path.append("../..")

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
from lung_function.modules.trans import LoadDatad, SaveDatad, RandomCropForegroundd, RemoveTextd
import os

PAD_DONE = False

def xformd(mode, z_size: int = 192, y_size: int = 256, x_size: int = 256, pad_truncated_dir='tmp', target='FVC', crop_foreground=False, pad_ratio=1, inputmode=None):
    post_pad_size = [int(i * pad_ratio) for i in [z_size, y_size, x_size]]
    
    if inputmode=='vessel':
        keys = ('vessel', )
        min_value, max_value = 0, 1
    elif inputmode=='image':
        keys = ('image', )
        min_value, max_value = -1500, 1500
    else:
        raise Exception(f"wrong input mode: {inputmode}")
    if crop_foreground:
        keys = keys + ('lung_mask',)
    global PAD_DONE
    if not PAD_DONE or not os.path.isdir(pad_truncated_dir):
        PAD_DONE = False
        if not os.path.isdir(pad_truncated_dir):
            os.makedirs(pad_truncated_dir)
        xforms = [LoadDatad(keys=keys[0], target=target, crop_foreground=crop_foreground), AddChanneld(keys=keys)]
        xforms.append(SpatialPadd(keys=keys[0], spatial_size=post_pad_size, mode='constant', constant_values=min_value))
        if crop_foreground:
            xforms.append(SpatialPadd(keys=keys[1], spatial_size=post_pad_size, mode='constant', constant_values= 0))
        xforms.append(ScaleIntensityRanged(keys=keys[0], a_min=min_value, a_max=max_value, b_min=-1, b_max=1, clip=True))
        xforms.append(SaveDatad(keys=keys[0], pad_truncated_dir=pad_truncated_dir, crop_foreground=crop_foreground))
    else:
        xforms = [LoadDatad(target=target, crop_foreground=crop_foreground), AddChanneld(keys=keys)]
    # xforms.append()
    if mode == 'train':
        if crop_foreground:
            xforms.extend([RandomCropForegroundd(keys=keys, roi_size=[z_size, y_size, x_size], source_key='lung_mask')])
        else:
            xforms.extend([RandSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size], random_center=True, random_size=False)])
        # xforms.extend([RandGaussianNoised(keys=keys, prob=0.5, mean=0, std=0.01)])
    else:
            xforms.extend([CenterSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size])])

    # xforms.append(SaveDatad(pad_truncated_dir+"/patches_examples/" + mode))

    # ('pat_id', 'image', 'lung_mask', 'origin', 'spacing', 'label')
    xforms.extend([CastToTyped(keys = keys, dtype=np.float32),
                   ToTensord(keys = keys),
                   RemoveTextd(keys='fpath')])
    transform = monai.transforms.Compose(xforms)

    return transform

def clean_data(pft_df, data_dir):
    pft_df.drop(pft_df[np.isnan(pft_df.DLCO_SB)].index, inplace=True)
    pft_df.drop(pft_df[pft_df.DLCO_SB == 0].index, inplace=True)
    pft_df.drop(pft_df[np.isnan(pft_df['FEV1'])].index, inplace=True)
    pft_df.drop(pft_df[pft_df['FEV1'] == 0].index, inplace=True)
    pft_df.drop(pft_df[np.isnan(pft_df.DateDF_abs)].index, inplace=True)
    pft_df.drop(pft_df[pft_df.DateDF_abs > 10].index, inplace=True)
    pft_df.drop(pft_df[pft_df['DLCOc/pred'] == "NV"].index, inplace=True)
    pft_df.drop(pft_df[pft_df['FVC/predNew'] == "NV"].index, inplace=True)

    scans = glob.glob(data_dir + "/SSc_patient_???????_GcVessel.nii.gz")  # get availabel files


    availabel_id_set = set([Path(id).stem[:19] for id in scans if not (('0422335' in id) 
    or ('0456204' in id) or ('6216732' in id) or ('6318989' in id))])  # exclude '422335' which has bad image quality

    pft_df.drop(pft_df.loc[~pft_df['subjectID'].isin(availabel_id_set)].index, inplace=True)


    # pft_df = pft_df.drop(pft_df[pft_df['subjectID'] not in availabel_id_set].index)
    # print(f"length of scans: {len(scans)}, length of labels: {len(pft_df)}")
    assert len(scans)>=len(pft_df)


    return pft_df

def pat_fromo_csv(mode: str, data) -> np.ndarray:
    tmp_ls = []
    df = pd.read_csv(f"/data1/jjia/lung_function/lung_function/scripts/results/experiments/914/{mode}_label.csv")
    pat_ls = [patid for patid in df['pat_id']]
    for d in data:
        if int(d['subjectID'].split('_')[-1]) in pat_ls:
            tmp_ls.append(d)
    tmp_np = np.array(tmp_ls)
    return tmp_np
    
def all_loaders(data_dir, label_fpath, args, datasetmode=('train', 'valid', 'test'), nb=None):

    pad_truncated_dir = f"/home/jjia/data/dataset/lung_function/iso{args.ct_sp}/z{args.z_size}x{args.x_size}y{args.y_size}_pad_ratio{str(args.pad_ratio)}"

    label_excel = pd.read_excel(label_fpath, engine='openpyxl')
    label_excel = label_excel.sort_values(by=['subjectID'])

    label_excel = clean_data(label_excel, data_dir)




    # 3 labels for one level
    data = np.array(label_excel.to_dict('records'))  # nparray is easy for kfold split
    for d in data:
        if not PAD_DONE or not os.path.isdir(pad_truncated_dir):
            if args.input_mode=="vessel":
                d['fpath'] = data_dir + '/' + d['subjectID'] + '_GcVessel.nii.gz'
            elif args.input_mode=="ct":
                d['fpath'] = data_dir + '/' + d['subjectID'] + '.nii.gz'
            else:
                raise Exception(f"wrong input mode: {args.input_mode}")
        else:
            if args.input_mode=="vessel":
                d['fpath'] = pad_truncated_dir + '/' + d['subjectID'] + '_GcVessel.nii.gz'
            elif args.input_mode=="ct":
                d['fpath'] = pad_truncated_dir + '/' + d['subjectID'] + '.nii.gz'
            else:
                raise Exception(f"wrong input mode: {args.input_mode}")

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

    # random.shuffle(data)  # Four fold are not right !!!
    if args.test_pat == 'random_as_ori':
        
        tr_data = pat_fromo_csv('train', data)
        vd_data = pat_fromo_csv('valid', data)
        ts_data = pat_fromo_csv('test', data)
    else:
        kf = KFold(n_splits=args.total_folds, shuffle=True, random_state=args.kfold_seed)  # for future reproduction

        if hasattr(args, 'test_pat') and args.test_pat == 'zhiwei77':
            ts_pat_ids = ['9071115', '6503304', '6587088', '7852072', '0911478', '5112278', '9075756', '4125990', '0584534',
    '4945176', '3034278', '2712128', '1043946', '9934096', '5240010', '7135410', '7421048', '9367440', '5576984', '0152440',
    '3154090', '1160750', '6484444', '1105441', '4628660', '4171220', '1146160', '2131790', '0163750', '2151769', '5174713',
    '8365740', '2524918', '9239682', '3243752', '2341332', '7234834', '9160660', '5262908', '2253442', '0992750', '3567342',
    '5271048', '8278747', '9662556', '0222357', '8229975', '0139552', '0458234', '9241693', '3901150', '9300979', '0298877',
    '3228438', '8960279', '4107789', '7740702', '7252792', '8303176', '8492153', '5299407', '7957098', '1499510', '5323286',
    '5325396', '3310402', '5813928', '6122288', '0315573', '2346390', '5869896', '0280727', '5352138', '8353193', '5321814',
    '6329587', '1397732',]
            ts_data, tr_vd_data = [], []
            for d in data:
                if d['subjectID'].split('_')[-1] in ts_pat_ids:
                    ts_data.append(d)
                else:
                    tr_vd_data.append(d)
            ts_data = np.array(ts_data)
            tr_vd_data = np.array(tr_vd_data)
            print(f"length of testing data: {len(ts_data)}")
        else:
            ts_nb = int(0.2 * len(data))
            tr_vd_data, ts_data = data[:-ts_nb], data[-ts_nb:]
        kf_list = list(kf.split(tr_vd_data))
        tr_pt_idx, vd_pt_idx = kf_list[args.fold - 1]
        tr_data = tr_vd_data[tr_pt_idx]
        vd_data = tr_vd_data[vd_pt_idx]
        print(f"length of training data: {len(tr_data)}")
    if nb:
        tr_data, vd_data, ts_data = tr_data[:nb], vd_data[:nb], ts_data[:nb]
    # tr_data, vd_data, ts_data = tr_data[:10], vd_data[:10], ts_data[:10]
    for d in [tr_data, vd_data, ts_data]:
        print(f"-----")
        for d_one in d:
            print(d_one['subjectID'])
    # trxformd = xformd('train')
    # vdxformd = xformd('valid')
    # tsxformd = xformd('test')
    data_dt = {}
    if 'train' in datasetmode:
        tr_dataset = monai.data.CacheDataset(data=tr_data, transform=xformd('train', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, pad_truncated_dir=pad_truncated_dir, target=args.target, crop_foreground=args.crop_foreground, pad_ratio=args.pad_ratio, inputmode=args.input_mode), num_workers=0, cache_rate=1)
        train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True)
        data_dt['train'] = train_dataloader

    if 'valid' in datasetmode:
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=xformd('valid', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, pad_truncated_dir=pad_truncated_dir, target=args.target, crop_foreground=args.crop_foreground, pad_ratio=args.pad_ratio, inputmode=args.input_mode), num_workers=0, cache_rate=1)
        valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)
        data_dt['valid'] = valid_dataloader

    if 'test' in datasetmode:
        ts_dataset = monai.data.CacheDataset(data=ts_data, transform=xformd('test', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, pad_truncated_dir=pad_truncated_dir, target=args.target, crop_foreground=args.crop_foreground, pad_ratio=args.pad_ratio, inputmode=args.input_mode), num_workers=0, cache_rate=1)
        test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)
        data_dt['test'] = test_dataloader

    # if 'trainnoaug' in datasetmode:
    #     # training dataset without any data augmentation to simulate the valid transform to see if center crop helps
    #     tr_dataset_no_aug = monai.data.CacheDataset(data=tr_data, transform=xformd('valid', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, pad_truncated_dir=pad_truncated_dir, target=args.target, crop_foreground=args.crop_foreground, pad_ratio=args.pad_ratio), num_workers=0, cache_rate=1)
    #     train_dataloader_no_aug = DataLoader(tr_dataset_no_aug, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)
    #     data_dt['trainnoaug'] = train_dataloader_no_aug

    return data_dt