# -*- coding: utf-8 -*-
# @Time    : 7/11/21 2:31 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import itertools
import json
import os
from lung_function.modules.trans import LoadDatad, SaveDatad, RandomCropForegroundd, RemoveTextd, LoadPointCloud, SampleShuffled
from sklearn.model_selection import KFold
import monai
from torch.utils.data import Dataset
import pandas as pd
from monai.transforms import RandSpatialCropd, RandGaussianNoised, CastToTyped, ToTensord, \
    CenterSpatialCropd, AddChanneld, ScaleIntensityRanged, SpatialPadd
from monai.data import DataLoader
from tqdm import tqdm
import torch
from mlflow import log_metric, log_param, log_params
import numpy as np
from pathlib import Path
from medutils.medutils import load_itk
import random
import glob
import sys
sys.path.append("../..")


# import streamlit as st


PAD_DONE = False


def build_dataset(file_ls, PNB=140000):
    points_ls = []
    for i in file_ls:
        a = pd.read_pickle(i)
        # convert voxel location to physical mm
        b = a['data'][:PNB, :3] * a['spacing']
        c = np.concatenate((b, a['data'][:PNB, -1].reshape(-1, 1)), axis=1)
        points_ls.append(c)
    points_np = np.array(points_ls)
    return points_np


def xformd(mode, args, pad_truncated_dir='tmp'):
    z_size = args.z_size
    y_size = args.y_size
    x_size = args.x_size
    target = args.target
    crop_foreground = args.crop_foreground
    pad_ratio = args.pad_ratio
    inputmode = args.input_mode
    PNB = args.PNB

    post_pad_size = [int(i * pad_ratio) for i in [z_size, y_size, x_size]]
    keys = (inputmode, )
    if inputmode == 'vessel_skeleton_pcd':
        xforms = [LoadPointCloud(keys=keys, target=target, position_center_norm=args.position_center_norm, PNB=PNB, repeated_sample=args.repeated_sample),
                #   SampleShuffled(
                #       keys=keys, PNB=PNB, repeated_sample=args.repeated_sample),
                  # ShiftCoordinated(keys=keys, position_center_norm=args.position_center_norm),
                  CastToTyped(keys=keys, dtype=np.float32),
                  ToTensord(keys=keys),
                  RemoveTextd(keys='fpath')]

    else:
        if inputmode == 'vessel':
            min_value, max_value = 0, 1
        elif 'ct' in inputmode:
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
            xforms = [LoadDatad(keys=keys[0], target=target, crop_foreground=crop_foreground,
                                inputmode=inputmode), AddChanneld(keys=keys)]
            xforms.append(SpatialPadd(
                keys=keys[0], spatial_size=post_pad_size, mode='constant', constant_values=min_value))
            if crop_foreground:
                xforms.append(SpatialPadd(
                    keys=keys[1], spatial_size=post_pad_size, mode='constant', constant_values=0))
            xforms.append(ScaleIntensityRanged(
                keys=keys[0], a_min=min_value, a_max=max_value, b_min=-1, b_max=1, clip=True))
            # xforms.append(SaveDatad(
            #     keys=keys[0], pad_truncated_dir=pad_truncated_dir, crop_foreground=crop_foreground, inputmode=inputmode))
        else:  # TODO: not implemented yet
            xforms = [LoadDatad(
                target=target, crop_foreground=crop_foreground), AddChanneld(keys=keys)]
        # xforms.append()
        if mode == 'train':
            if crop_foreground:
                xforms.extend([RandomCropForegroundd(keys=keys, roi_size=[
                              z_size, y_size, x_size], source_key='lung_mask')])
            else:
                xforms.extend([RandSpatialCropd(keys=keys, roi_size=[
                              z_size, y_size, x_size], random_center=True, random_size=False)])
            # xforms.extend([RandGaussianNoised(keys=keys, prob=0.5, mean=0, std=0.01)])
        else:
            xforms.extend(
                [CenterSpatialCropd(keys=keys, roi_size=[z_size, y_size, x_size])])

        # xforms.append(SaveDatad(pad_truncated_dir+"/patches_examples/" + mode))

        # ('pat_id', 'image', 'lung_mask', 'origin', 'spacing', 'label')
        xforms.extend([CastToTyped(keys=keys, dtype=np.float32),
                       ToTensord(keys=keys),
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

    for name in pft_df.columns:
        if name!='PatID':
            pft_df[name].replace('', np.nan, inplace=True)  # exclude 3 rows with NV or empty cells
            pft_df[name].replace('NV', np.nan, inplace=True)

            pft_df.dropna(subset=[name], inplace=True)
            pft_df[name] = pft_df[name].astype(float)

    # get availabel files
    scans = glob.glob(data_dir + "/SSc_patient_???????_GcVessel.nii.gz")
    if len(scans) == 0:
        # get availabel files
        scans = glob.glob(data_dir + "/SSc_patient_???????.mha")

    availabel_id_set = set([Path(id).stem[:19] for id in scans if not (
        ('0422335' in id))])  # exclude '422335' which has bad image quality
    # 0456204 had a different file name, 6216723 and 6318939 are typo, I need to add them back !!!

    pft_df.drop(pft_df.loc[~pft_df['subjectID'].isin(
        availabel_id_set)].index, inplace=True)

    # pft_df = pft_df.drop(pft_df[pft_df['subjectID'] not in availabel_id_set].index)
    # print(f"length of scans: {len(scans)}, length of labels: {len(pft_df)}")
    assert len(scans) >= len(pft_df)

    return pft_df


def pat_fromo_csv(mode: str, data, fold=1) -> np.ndarray:
    tmp_ls = []
    ex_fold_dt = {1: '905', 2: '914', 3: '919', 4: '924'}
    df = pd.read_csv(
        f"/data1/jjia/lung_function/lung_function/scripts/results/experiments/{ex_fold_dt[fold]}/{mode}_label.csv")
    pat_ls = [patid for patid in df['pat_id']]
    for d in data:
        if int(d['subjectID'].split('_')[-1]) in pat_ls:
            tmp_ls.append(d)
    tmp_np = np.array(tmp_ls)
    return tmp_np


def pat_from_json(data, fold=1) -> np.ndarray:
    with open('/home/jjia/data/lung_function/lung_function/modules/data_split.json', "r") as f:
        data_split = json.load(f)

    valid = data_split[f'valid_fold{fold}']
    test = data_split[f'test']
    # train = []
    train = list(itertools.chain(
        *[data_split[f'valid_fold{i}'] for i in [1, 2, 3, 4] if i != fold]))
    # for i in tmp_ls:
    #     train.extend(i)

    def avail_data(pat_ls, data) -> np.ndarray:
        tmp_ls = []
        for d in data:
            tmp_id = d['subjectID'].split('_')[-1]
            if tmp_id in pat_ls:
                tmp_ls.append(d)
        return np.array(tmp_ls)

    train = avail_data(train, data)
    valid = avail_data(valid, data)
    test = avail_data(test, data)
    return train, valid, test


def all_loaders(data_dir, label_fpath, args, datasetmode=('train', 'valid', 'test'), nb=None):

    if args.ct_sp in ('1.0', '1.5'):
        pad_truncated_dir = f"/home/jjia/data/dataset/lung_function/iso{args.ct_sp}/z{args.z_size}x{args.x_size}y{args.y_size}_pad_ratio{str(args.pad_ratio)}"
    else:
        pad_truncated_dir = f"/home/jjia/data/dataset/lung_function/ori_resolution/z{args.z_size}x{args.x_size}y{args.y_size}_pad_ratio{str(args.pad_ratio)}"

    label_excel = pd.read_excel(label_fpath, engine='openpyxl')
    label_excel = label_excel.sort_values(by=['subjectID'])
    label_excel = clean_data(label_excel, data_dir)

    # 3 labels for one level
    # nparray is easy for kfold split
    data = np.array(label_excel.to_dict('records'))
    for d in data:
        if args.input_mode == 'vessel_skeleton_pcd':  # do not need to chare if padding or not
            d['fpath'] = data_dir + '/' + d['subjectID'] + \
                '_skeleton_coordinates140000.pt'
        else:
            if not PAD_DONE or not os.path.isdir(pad_truncated_dir):
                if args.input_mode == "vessel":
                    d['fpath'] = data_dir + '/' + \
                        d['subjectID'] + '_GcVessel.nii.gz'
                else:
                    d['fpath'] = data_dir + '/' + d['subjectID'] + '.nii.gz'

                    # # raise Exception(f"wrong input mode: {args.input_mode}")
                    # pass
            else:  # TODO: not implemented yet
                if args.input_mode == "vessel":
                    d['fpath'] = pad_truncated_dir + '/' + \
                        d['subjectID'] + '_GcVessel.nii.gz'
                elif args.input_mode == "ct":
                    d['fpath'] = pad_truncated_dir + \
                        '/' + d['subjectID'] + '.nii.gz'
                elif "ct_masked_by_vessel" in args.input_mode:
                    d['fpath'] = pad_truncated_dir + \
                        '/' + d['subjectID'] + '.nii.gz'
                else:
                    raise Exception(f"wrong input mode: {args.input_mode}")

    # random.shuffle(data)  # Four fold are not right !!!
    if args.test_pat == 'random_as_ori':
        tr_data, vd_data, ts_data = pat_from_json(data, args.fold)
    else:
        kf = KFold(n_splits=args.total_folds, shuffle=True,
                   random_state=args.kfold_seed)  # for future reproduction

        if hasattr(args, 'test_pat') and args.test_pat == 'zhiwei77':
            ts_pat_ids = ['9071115', '6503304', '6587088', '7852072', '0911478', '5112278', '9075756', '4125990',
                          '0584534', '4945176', '3034278', '2712128', '1043946', '9934096', '5240010', '7135410',
                          '7421048', '9367440', '5576984', '0152440', '3154090', '1160750', '6484444', '1105441',
                          '4628660', '4171220', '1146160', '2131790', '0163750', '2151769', '5174713', '8365740',
                          '2524918', '9239682', '3243752', '2341332', '7234834', '9160660', '5262908', '2253442',
                          '0992750', '3567342', '5271048', '8278747', '9662556', '0222357', '8229975', '0139552',
                          '3901150', '9300979', '0298877', '3228438', '8960279', '4107789', '7740702', '7252792',
                          '8303176', '8492153', '5299407', '7957098', '1499510', '5323286', '5325396', '3310402',
                          '5813928', '6122288', '0315573', '2346390', '5869896', '0280727', '5352138', '8353193',
                          '5321814', '6329587', '1397732',]
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
    # for d in [tr_data, vd_data, ts_data]:
    #     print("-----")
    #     for d_one in d:
    #         print(d_one['subjectID'])
    # trxformd = xformd('train')
    # vdxformd = xformd('valid')
    # tsxformd = xformd('test')
    data_dt = {}
    if 'train' in datasetmode:
        tr_dataset = monai.data.CacheDataset(data=tr_data, transform=xformd(
            'train', args, pad_truncated_dir=pad_truncated_dir), num_workers=0, cache_rate=1)
        train_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                      shuffle=True, num_workers=args.workers, persistent_workers=True)
        data_dt['train'] = train_dataloader

    if 'valid' in datasetmode:
        vd_dataset = monai.data.CacheDataset(data=vd_data, transform=xformd(
            'valid', args, pad_truncated_dir=pad_truncated_dir), num_workers=0, cache_rate=1)
        valid_dataloader = DataLoader(vd_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.workers, persistent_workers=True)
        data_dt['valid'] = valid_dataloader

    if 'test' in datasetmode:
        ts_dataset = monai.data.CacheDataset(data=ts_data, transform=xformd(
            'test', args, pad_truncated_dir=pad_truncated_dir), num_workers=0, cache_rate=1)
        test_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers, persistent_workers=True)
        data_dt['test'] = test_dataloader

    # if 'trainnoaug' in datasetmode:
    #     # training dataset without any data augmentation to simulate the valid transform to see if center crop helps
    #     tr_dataset_no_aug = monai.data.CacheDataset(data=tr_data, transform=xformd('valid', z_size=args.z_size, y_size=args.y_size, x_size=args.x_size, pad_truncated_dir=pad_truncated_dir, target=args.target, crop_foreground=args.crop_foreground, pad_ratio=args.pad_ratio), num_workers=0, cache_rate=1)
    #     train_dataloader_no_aug = DataLoader(tr_dataset_no_aug, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True)
    #     data_dt['trainnoaug'] = train_dataloader_no_aug

    return data_dt
