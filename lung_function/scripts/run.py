# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
import random
import statistics
import threading
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from medutils import medutils
from medutils.medutils import count_parameters
from mlflow import log_metric, log_metrics, log_param, log_params
from mlflow.tracking import MlflowClient
from monai.utils import set_determinism
from typing import List, Sequence
from argparse import Namespace
import functools
import thop
import os
import copy

from lung_function.modules import provider
from lung_function.modules.compute_metrics import icc, metrics
from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, dec_record_cgpu, retrive_run
import sys
sys.path.append("../modules/networks/models_pcd")

args = get_args()
global_lock = threading.Lock()


def thread_safe(func):
    def thread_safe_fun(*args, **kwargs):
        with global_lock:
            print('get lock by main thread')
            func(*args, **kwargs)
            print('release lock by main thread')
    return thread_safe_fun


def try_func(func):
    def _try_fun(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as err:
            print(err, file=sys.stderr)
            pass
    return _try_fun


log_metric = try_func(log_metric)
log_metrics = try_func(log_metrics)


def reinit_fc(net, nb_fc0, fc1_nodes, fc2_nodes, num_classes):
    net.ln1 = nn.Linear(nb_fc0, fc1_nodes)
    net.rl1 = nn.ReLU(inplace=True)
    net.dp1 = nn.Dropout()
    net.ln2 = nn.Linear(fc1_nodes, fc2_nodes)
    net.rl2 = nn.ReLU(inplace=True)
    net.dp2 = nn.Dropout()
    net.ln3 = nn.Linear(fc2_nodes, num_classes)
    return net


def int2str(batch_id: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        batch_id (np.ndarray): _description_

    Raises:
        Exception: _description_

    Returns:
        np.ndarray: _description_
    """
    tmp = batch_id.shape
    id_str_ls = []
    for id in batch_id:
        if isinstance(id, np.ndarray):
            id = id[0]
        id = str(id)
        while len(id) < 7:  # the pat id should be 7 digits
            id = '0' + id
        if len(tmp) == 2:
            id_str_ls.append([id])
        elif len(tmp) == 1:
            id_str_ls.append(id)
        else:
            raise Exception(
                f"the shape of batch_id is {tmp}, but it should be 1-dim or 2-dim")

    return np.array(id_str_ls)


class Run:
    """A class which has its dataloader and step_iteration. It is like Lighting. 
    """

    def __init__(self, args: Namespace):
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        self.device = torch.device("cuda")  # 'cuda'
        self.target = [i.lstrip() for i in args.target.split('-')]

        self.pointnet_fc_ls = [int(i) for i in args.pointnet_fc_ls.split('-')]
        self.net = get_net_3d(name=args.net, nb_cls=len(self.target), image_size=args.x_size,
                              pretrained=args.pretrained_imgnet, pointnet_fc_ls=self.pointnet_fc_ls, loss=args.loss,
                              dp_fc1_flag = args.dp_fc1_flag, args=args)  # output FVC and FEV1
        self.fold = args.fold
        self.flops_done = False

        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1e6, 2))
        log_param('net_parameters_M', net_parameters)

        self.data_dt = all_loaders(
            self.mypath.data_dir, self.mypath.label_fpath, args, nb=2)
        self.loss_fun = get_loss(
            args.loss, mat_diff_loss_scale=args.mat_diff_loss_scale)
        self.opt = torch.optim.Adam(
            self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.net = self.net.to(self.device)

        validMAEEpoch_AllBest = 1000
        if args.pretrained_id != '0':
            if 'SSc' in args.pretrained_id:  # pretrained by ssc_pos L-Net weights
                pretrained_id = args.pretrained_id.split(
                    '-')[self.fold]  # [852] [853] [854] [855]
                pretrained_model_path = f"/home/jjia/data/ssc_scoring/ssc_scoring/results/models_pos/{pretrained_id}/model.pt"
                print(f"pretrained_model_path: {pretrained_model_path}")
                ckpt = torch.load(pretrained_model_path,
                                  map_location=self.device)
                del ckpt['ln3.weight']
                del ckpt['ln3.bias']
                del self.net.ln3  # remove the last layer because they do not match

                # model_fpath need to exist
                self.net.load_state_dict(ckpt, strict=False)
                self.net = reinit_fc(self.net, nb_fc0=8 * 16 * 6 * 6 * 6, fc1_nodes=1024, fc2_nodes=1024,
                                     num_classes=len(self.target))
                # move the new initialized layers to GPU
                self.net = self.net.to(self.device)
                print(f"use the pretrained model from {pretrained_model_path}")

            else:
                pretrained_path = PFTPath(
                    args.pretrained_id, check_id_dir=False, space=args.ct_sp)
                ckpt = torch.load(pretrained_path.model_fpath,
                                  map_location=self.device)

                if type(ckpt) is dict and 'model' in ckpt:
                    model = ckpt['model']
                    if 'metric_name' in ckpt:
                        if 'validMAEEpoch_AllBest' == ckpt['metric_name']:
                            validMAEEpoch_AllBest = ckpt['current_metric_value']
                else:
                    model = ckpt
                # model_fpath need to exist
                self.net.load_state_dict(model, strict=False)
                # move the new initialized layers to GPU
                self.net = self.net.to(self.device)

        self.BestMetricDt = {'trainLossEpochBest': 1000,
                             # 'trainnoaugLossEpochBest': 1000,
                             'validLossEpochBest': 1000,
                             'testLossEpochBest': 1000,

                             'trainMAEEpoch_AllBest': 1000,
                             # 'trainnoaugMAEEpoch_AllBest': 1000,
                             'validMAEEpoch_AllBest': validMAEEpoch_AllBest,
                             'testMAEEpoch_AllBest': 1000,
                             }

    def step(self, mode, epoch_idx, save_pred=False):
        dataloader = self.data_dt[mode]
        loss_fun_mae = nn.L1Loss()

        scaler = torch.cuda.amp.GradScaler()
        print(mode + "ing ......")
        if mode == 'train':
            self.net.train()
        else:
            self.net.eval()

        t0 = time.time()
        data_idx = 0
        loss_accu = 0
        mae_accu_ls = [0 for _ in self.target]
        mae_accu_all = 0
        for data in dataloader:
            torch.cuda.empty_cache()  # avoid memory leak
            data_idx += 1
            if epoch_idx < 3:  # only show first 3 epochs' data loading time
                t1 = time.time()
                log_metric('TLoad', t1 - t0, data_idx + epoch_idx * len(dataloader))
            key = args.input_mode

            if args.input_mode == 'vessel_skeleton_pcd':
                points = data[key].data.numpy()
                if points.shape[0] == 1:  # batch size=1
                    points = np.concatenate([points, points])
                    data['label'] = np.concatenate(
                        [data['label'], data['label']])
                    data['label'] = torch.tensor(data['label'])

                points = provider.random_point_dropout(points)
                # points[:, :, 0:3] = provider.random_scale_point_cloud(
                #     points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(
                    points[:, :, 0:3], shift_range=args.shift_range)
                points = torch.Tensor(points)
                points = points.transpose(2, 1)
                data[key] = points

            batch_x = data[key]  # n, z, y, x
            if args.input_mode == 'ct_left':  # 2 is left
                a = copy.deepcopy(data['lung_mask'])
                a[a!=2] = 0
                batch_x = batch_x * a
            elif args.inputmode == 'ct_right':  # 1 is right
                a = copy.deepcopy(data['lung_mask'])
                a[a!=1] = 0
                batch_x = batch_x * a
            else:
                pass
            batch_x = batch_x.to(self.device)  # n, z, y, x
            batch_y = data['label'].to(self.device)

            if not self.flops_done:  # only calculate teh macs and params once
                macs, params = thop.profile(self.net, inputs=(batch_x, ))
                self.flops_done = True
                log_param('macs_G', str(round(macs/1e9, 2)))
                log_param('net_params_M', str(round(params/1e6, 2)))

            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        if args.loss == 'mse_regular':
                            pred, trans_feat = self.net(batch_x)
                        else:
                            pred = self.net(batch_x)
                else:
                    if args.loss == 'mse_regular':
                        pred, trans_feat = self.net(batch_x)
                    else:
                        pred = self.net(batch_x)
                if save_pred:
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = data['pat_id'].cpu(
                    ).detach().numpy()  # shape (N,1)
                    batch_pat_id = int2str(batch_pat_id)  # shape (N,1)

                    batch_y_np = batch_y.cpu().detach().numpy()  # shape (N, out_nb)
                    pred_np = pred.cpu().detach().numpy()  # shape (N, out_nb)
                    # batch_pat_id = np.expand_dims(batch_pat_id, axis=-1)  # change the shape from (N,) to (N, 1)

                    # shape (1,1)
                    if args.input_mode == 'vessel_skeleton_pcd' and len(batch_pat_id) == 1:
                        batch_pat_id = np.array(
                            [[int(batch_pat_id[0])], [int(batch_pat_id[0])]])
                        batch_pat_id = torch.tensor(batch_pat_id)

                    saved_label = np.hstack((batch_pat_id, batch_y_np))
                    saved_pred = np.hstack((batch_pat_id, pred_np))
                    medutils.appendrows_to(self.mypath.save_label_fpath(
                        mode), saved_label, head=head)
                    medutils.appendrows_to(
                        self.mypath.save_pred_fpath(mode), saved_pred, head=head)

                if args.loss == 'mse_regular':
                    loss = self.loss_fun(pred, batch_y, trans_feat)
                else:
                    loss = self.loss_fun(pred, batch_y)
                with torch.no_grad():
                    mae_ls = [loss_fun_mae(pred[:, i], batch_y[:, i]).item()
                              for i in range(len(self.target))]
                    mae_all = loss_fun_mae(pred, batch_y).item()

            if mode == 'train' and save_pred is not True:  # update gradients only when training
                self.opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt)
                scaler.update()
            loss_cpu = loss.item()
            print('loss:', loss_cpu)
            # log_metric(mode+'LossBatch', loss_cpu, data_idx+epoch_idx*len(dataloader))
            # log_metric(mode+'MAEBatch_All', mae_all, data_idx+epoch_idx*len(dataloader))
            # [log_metric(mode+'MAEBatch_'+t, m, data_idx+epoch_idx*len(dataloader)) for t, m in zip(self.target, mae_ls)]

            loss_accu += loss_cpu
            for i, mae in enumerate(mae_ls):
                mae_accu_ls[i] += mae
            mae_accu_all += mae_all

            # print('pred:', pred.clone().detach().cpu().numpy())
            # print('label:', batch_y.clone().detach().cpu().numpy())
            if epoch_idx < 3:
                t2 = time.time()
                log_metric('TUpdateWBatch', t2-t1, data_idx +
                           epoch_idx*len(dataloader))
                t0 = t2  # reset the t0
        log_metric(mode+'LossEpoch', loss_accu/len(dataloader), epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all /
                   len(dataloader), epoch_idx)
        for t, i in zip(self.target, mae_accu_ls):
            log_metric(mode + 'MAEEpoch_' + t, i / len(dataloader), epoch_idx)

        self.BestMetricDt[mode + 'LossEpochBest'] = min(
            self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len(dataloader))
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min(
            self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len(dataloader))

        log_metric(mode+'LossEpochBest',
                   self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest',
                   self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len(dataloader):
            for t, i in zip(self.target, mae_accu_ls):
                log_metric(mode + 'MAEEpoch_' + t + 'Best',
                           i / len(dataloader), epoch_idx)

            if mode == 'valid':
                print(
                    f"Current mae is {self.BestMetricDt[mode+'MAEEpoch_AllBest']}, better than the previous mae: {tmp}, save model.")
                ckpt = {'model': self.net.state_dict(),
                        'metric_name': mode+'MAEEpoch_AllBest',
                        'current_metric_value': self.BestMetricDt[mode+'MAEEpoch_AllBest']}
                torch.save(ckpt, self.mypath.model_fpath)


@dec_record_cgpu(args.outfile)
def run(args: Namespace):
    """
    Run the whole  experiment using this args.
    """
    myrun = Run(args)
    modes = ['train', 'valid', 'test']
    if args.mode == 'infer':
        for mode in ['valid', 'test']:
            myrun.step(mode,  0,  save_pred=True)
    else:  # 'train' or 'continue_train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation
                myrun.step('valid',  i)
                myrun.step('test',  i)
            if i == args.epochs - 1:  # load best model and do inference
                print('start inference')
                if os.path.exists(myrun.mypath.model_fpath):
                    ckpt = torch.load(myrun.mypath.model_fpath,
                                      map_location=myrun.device)
                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        model = ckpt['model']
                    else:
                        model = ckpt
                    # model_fpath need to exist
                    myrun.net.load_state_dict(model)
                    print(f"load net from {myrun.mypath.model_fpath}")
                else:
                    print(
                        f"no model found at {myrun.mypath.model_fpath}, let me save the current model to this lace")
                    ckpt = {'model': myrun.net.state_dict()}
                    torch.save(ckpt, myrun.mypath.model_fpath)
                for mode in modes:
                    myrun.step(mode, i, save_pred=True)

    mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
    label_ls = [mypath.save_label_fpath(mode) for mode in modes]
    pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

    for pred_fpath, label_fpath in zip(pred_ls, label_ls):
        r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True)
        log_params(r_p_value)
        print('r_p_value:', r_p_value)

        icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
        log_params(icc_value)
        print('icc:', icc_value)

    print('Finish all things!')


def average_all_folds(id_ls: Sequence[int], current_id: int, experiment, key='params'):
    """
    Average the logs form mlflow for all folds.
    """
    current_run = retrive_run(experiment=experiment, reload_id=current_id)

    all_dt = {}
    for id in id_ls:
        mlflow_run = retrive_run(experiment=experiment, reload_id=id)
        if key == 'params':
            target_dt = mlflow_run.data.params
            current_dt = current_run.data.params
        elif key == 'metrics':
            target_dt = mlflow_run.data.metrics
            current_dt = current_run.data.metrics
        else:
            raise Exception(
                f"Expected key of 'params' or 'metrics', but got key: {key}")

        for k, v in target_dt.items():
            if k not in current_dt:  # re-writing parameters in mlflow is not allowed
                if k not in all_dt:
                    all_dt[k] = []
                # this is a value, not a list (see bellow)
                if not isinstance(all_dt[k], list):
                    continue
                try:
                    all_dt[k].append(float(v))
                except Exception:
                    # can not be converted to numbers which can not be averaged
                    all_dt[k] = v

    all_dt = {k: statistics.mean(v) if isinstance(
        v, list) else v for k, v in all_dt.items()}

    return all_dt


def log_metrics_all_folds_average(id_ls: list, id: int, experiment):
    """
    Get the 4 folds metrics and parameters
    Average them
    Log average values to the parent mlflow
    """
    # average parameters
    param_dt = average_all_folds(id_ls, id, experiment, key='params')
    if len(param_dt) < 100:
        log_params(param_dt)

    elif len(param_dt) >= 100 and len(param_dt) < 200:
        dt_1 = {k: param_dt[k] for i, k in enumerate(param_dt) if i < 100}
        dt_2 = {k: param_dt[k] for i, k in enumerate(param_dt) if i >= 100}
        log_params(dt_1)
        log_params(dt_2)
    else:
        raise Exception(
            f"Our logging request can contain at most 200 params. Got {len(param_dt)} params")

    # average metrics
    metric_dt = average_all_folds(id_ls, id, experiment, key='metrics')
    log_metrics(metric_dt, 0)


def main():
    SEED = 4
    set_determinism(SEED)  # set seed for this run

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)

    random.seed(SEED)
    np.random.seed(SEED)

    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    RECORD_FPATH = "results/record.log"
    # write super parameters from set_args.py to record file.
    id = record_1st(RECORD_FPATH)

    # if merge 4 fold results, uncommit the following code.
    # From here ======================================================
    # current_id = 427
    # id_ls = [428, 431, 433, 435]
    # client = MlflowClient()
    # run_ls = client.search_runs(experiment_ids=[experiment.experiment_id],
    #                             filter_string=f"params.id LIKE '%{current_id}%'")
    # run_ = run_ls[0]
    # run_id = run_.info.run_id
    # with mlflow.start_run(run_id=run_id, tags={"mlflow.note.content": args.remark}):
    #     args.id = id  # do not need to pass id seperately to the latter function

    # to here =======================================================
    if args.mode == 'infer':  # get the id of the run
        client = MlflowClient()
        run_ls = client.search_runs(experiment_ids=[experiment.experiment_id], run_view_type=3,  # run_view_type=2 means the 'deleted'
                                    filter_string=f"params.id LIKE '%{args.pretrained_id}%'")
        run_ = run_ls[0]
        run_id = run_.info.run_id
        with mlflow.start_run(run_id=run_id, tags={"mlflow.note.content": args.remark}):
            args.id = args.pretrained_id  # log the metrics to the pretrained_id
            run(args)
    else:
        with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
            args.id = id  # do not need to pass id seperately to the latter function

            current_id = id
            tmp_args_dt = vars(args)
            tmp_args_dt['fold'] = 'all'
            log_params(tmp_args_dt)

            if '-' in args.pretrained_id:
                pretrained_ids = args.pretrained_id.split('-')
            else:
                pretrained_ids = []

            all_folds_id_ls = []
            for fold in [1, 2, 3, 4]:
                # write super parameters from set_args.py to record file.
                if len(pretrained_ids):
                    args.pretrained_id = pretrained_ids[fold-1]

                id = record_1st(RECORD_FPATH)
                all_folds_id_ls.append(id)
                with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                    args.fold = fold

                    args.id = id  # do not need to pass id seperately to the latter function
                    tmp_args_dt = vars(args)
                    log_params(tmp_args_dt)
                    run(args)
            log_metrics_all_folds_average(
                all_folds_id_ls, current_id, experiment)


if __name__ == "__main__":
    main()
