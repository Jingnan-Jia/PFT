# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
from typing import List

sys.path.append("../..")
from mlflow import log_metric, log_metrics, log_param, log_params
import mlflow
import threading
import time
from medutils.medutils import count_parameters
from medutils import medutils
from queue import Queue
import torch
import torch.nn as nn
import copy
import statistics
from mlflow.tracking import MlflowClient
import numpy as np

from monai.utils import set_determinism

from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info, retrive_run
from lung_function.modules.compute_metrics import icc, metrics
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


def int2str(batch_id: np.ndarray) -> np.ndarray:
    id_str_ls = []
    for id in batch_id:
        id = str(id)
        while len(id) < 7:  # the pat id should be 7 digits
            id = '0' + id
        id_str_ls.append(id)
    return np.array(id_str_ls)

class Run:
    def __init__(self, args):
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        self.device = torch.device("cuda")
        self.target = [i.lstrip() for i in args.target.split('-')]
        self.net = get_net_3d(name=args.net, nb_cls=len(self.target), image_size=args.x_size) # output FVC and FEV1
        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1000 / 1000, 2))
        log_param('net_parameters_M', net_parameters)

        self.data_dt = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args)

        self.loss_fun = get_loss(args.loss)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.net = self.net.to(self.device)

        validMAEEpoch_AllBest = 1000
        if args.pretrained_id:
            pretrained_path = PFTPath(args.pretrained_id, check_id_dir=False, space=args.ct_sp)
            ckpt = torch.load(pretrained_path.model_fpath, map_location=self.device)
            if type(ckpt) is dict and 'model' in ckpt:
                model = ckpt['model']
                if 'validMAEEpoch_AllBest' in ckpt:
                    validMAEEpoch_AllBest = ckpt['validMAEEpoch_AllBest']
            else:
                model = ckpt
            self.net.load_state_dict(model)  # model_fpath need to exist


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
        mae_accu_ls = [0 for i in self.target]
        mae_accu_all = 0
        for data in dataloader:
            data_idx += 1
            if epoch_idx < 3:  # only show first 3 epochs' data loading time
                t1 = time.time()
                log_metric('TLoad', t1-t0, data_idx+epoch_idx*len(dataloader))

            batch_x = data['image'].to(self.device)
            batch_y = data['label'].to(self.device)
            with torch.cuda.amp.autocast():
                if mode != 'train' or save_pred:  # save pred for inference
                    with torch.no_grad():
                        pred = self.net(batch_x)

                else:
                    pred = self.net(batch_x)
                if save_pred:
                    head = ['pat_id']
                    head.extend(self.target)

                    batch_pat_id = data['pat_id'].cpu().detach().numpy()
                    batch_pat_id = int2str(batch_pat_id)

                    batch_y_np = batch_y.cpu().detach().numpy()
                    pred_np = pred.cpu().detach().numpy()

                    batch_pat_id = np.expand_dims(batch_pat_id, axis=-1)  # change the shape from (N,) to (N, 1)

                    saved_label = np.hstack((batch_pat_id, batch_y_np))
                    saved_pred = np.hstack((batch_pat_id, pred_np))
                    medutils.appendrows_to(self.mypath.save_label_fpath(mode), saved_label, head=head)
                    medutils.appendrows_to(self.mypath.save_pred_fpath(mode), saved_pred, head=head)

                loss = self.loss_fun(pred, batch_y)
                with torch.no_grad():
                    mae_ls = [loss_fun_mae(pred[:, i], batch_y[:, i]).item() for i in range(len(self.target))]
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
                log_metric('TUpdateWBatch', t2-t1, data_idx+epoch_idx*len(dataloader))
                t0 = t2  # reset the t0
        log_metric(mode+'LossEpoch', loss_accu/len(dataloader), epoch_idx)
        log_metric(mode+'MAEEpoch_All', mae_accu_all/len(dataloader), epoch_idx)
        No = [log_metric(mode + 'MAEEpoch_' + t, i / len(dataloader), epoch_idx) for t, i in zip(self.target, mae_accu_ls)]

        self.BestMetricDt[mode + 'LossEpochBest'] = min(self.BestMetricDt[mode+'LossEpochBest'], loss_accu/len(dataloader))
        tmp = self.BestMetricDt[mode+'MAEEpoch_AllBest']
        self.BestMetricDt[mode + 'MAEEpoch_AllBest'] = min(self.BestMetricDt[mode+'MAEEpoch_AllBest'], mae_accu_all/len(dataloader))

        log_metric(mode+'LossEpochBest', self.BestMetricDt[mode + 'LossEpochBest'], epoch_idx)
        log_metric(mode+'MAEEpoch_AllBest', self.BestMetricDt[mode + 'MAEEpoch_AllBest'], epoch_idx)

        if self.BestMetricDt[mode+'MAEEpoch_AllBest'] == mae_accu_all/len(dataloader):
            [log_metric(mode + 'MAEEpoch_' + t +'Best', i / len(dataloader), epoch_idx) for t, i in zip(self.target, mae_accu_ls)]

            if mode == 'valid':
                print(f"Current mae is {self.BestMetricDt[mode+'MAEEpoch_AllBest']}, better than the previous mae: {tmp}, save model.")
                ckpt = {'model': self.net.state_dict(),
                        'metric_name': mode+'MAEEpoch_AllBest',
                        'current_metric_value': self.BestMetricDt[mode+'MAEEpoch_AllBest']}
                torch.save(ckpt, self.mypath.model_fpath)


def run(args):
    myrun = Run(args)
    infer_modes = ['train', 'valid', 'test']
    if args.mode == 'infer':
        for mode in infer_modes:
            myrun.step(mode,  0,  save_pred=True)
    else: # 'train' or 'continue_train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation
                myrun.step('valid',  i)
                myrun.step('test',  i)
            if i == args.epochs - 1:  # load best model and do inference
                print('start inference')
                ckpt = torch.load(myrun.mypath.model_fpath, map_location=myrun.device)
                if type(ckpt) is dict and 'model' in ckpt:
                    model = ckpt['model']
                else:
                    model = ckpt
                myrun.net.load_state_dict(model)  # model_fpath need to exist
                print(f"load net from {myrun.mypath.model_fpath}")
                for mode in infer_modes:
                    myrun.step(mode, i, save_pred=True)


    mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
    modes = infer_modes
    label_ls = [mypath.save_label_fpath(mode) for mode in modes]
    pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

    for pred_fpath, label_fpath in zip(pred_ls, label_ls):
        r_p_value = metrics(pred_fpath, label_fpath)
        log_params(r_p_value)
        print('r_p_value:', r_p_value)

        icc_value = icc(label_fpath, pred_fpath)
        log_params(icc_value)
        print('icc:', icc_value)

    print('Finish all things!')


def average_all_folds(id_ls, current_id, key='params'):
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
            raise Exception(f"Expected key of 'params' or 'metrics', but got key: {key}")

        for k, v in target_dt.items():
            if k not in current_dt:  # re-writing parameters in mlflow is not allowed
                if k not in all_dt:
                    all_dt[k] = []
                if type(all_dt[k]) is not list:  # this is a value, not a list (see bellow)
                    continue

                try:
                    all_dt[k].append(float(v))
                except:
                    all_dt[k] = v  # can not be converted to numbers which can not be averaged

    all_dt = {k: statistics.mean(v) if type(v) is list else v for k, v in all_dt.items() }

    return all_dt

def log_metrics_all_folds_average(id_ls, id):
    """
    Get the 4 folds metrics and parameters
    Average them
    Log average values to the parent mlflow
    """
    # average metrics


    # average parameters
    param_dt = average_all_folds(id_ls, id, key='params')
    if len(param_dt) < 100:
        log_params(param_dt)

    elif len(param_dt) >= 100 and len(param_dt) < 200:
        dt_1 = {k:param_dt[k] for i, k in enumerate(param_dt) if i < 100}
        dt_2 = {k:param_dt[k] for i, k in enumerate(param_dt) if i >= 100}
        log_params(dt_1)
        log_params(dt_2)
    else:
        raise Exception(f"A batch logging request can contain at most 100 params. Got {len(param_dt)} params")
    metric_dt =average_all_folds(id_ls, id, key='metrics')
    log_metrics(metric_dt, 0)


if __name__ == "__main__":
    set_determinism(seed=4)  # set seed for this run

    mlflow.set_tracking_uri("http://nodelogin02:5000")
    experiment = mlflow.set_experiment("lung_fun_db15")
    record_fpath = "results/record.log"
    id = record_1st(record_fpath)  # write super parameters from set_args.py to record file.

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
    if args.mode=='infer':  # get the id of the run
        client = MlflowClient()
        run_ls = client.search_runs(experiment_ids=[experiment.experiment_id], run_view_type=2,
                                    filter_string=f"params.id LIKE '%{args.pretrained_id}%'")
        run_ = run_ls[0]
        run_id = run_.info.run_id
        with mlflow.start_run(run_id=run_id, tags={"mlflow.note.content": args.remark}):
            args.id = args.pretrained_id
            # import requests
            # requests.post()
            run(args)
    else:
        with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
            args.id = id  # do not need to pass id seperately to the latter function

            current_id = id

            p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile, ))
            p1.start()

            tmp_args_dt = vars(args)
            tmp_args_dt['fold'] = 'all'
            log_params(tmp_args_dt)

            id_ls = []
            for fold in [1, 2, 3, 4]:
                id = record_1st(record_fpath)  # write super parameters from set_args.py to record file.
                id_ls.append(id)
                with mlflow.start_run(run_name=str(id) + '_fold_' + str(fold), tags={"mlflow.note.content": f"fold: {fold}"}, nested=True):
                    args.fold = fold
                    args.id = id  # do not need to pass id seperately to the latter function
                    tmp_args_dt = vars(args)
                    log_params(tmp_args_dt)
                    run(args)

            log_metrics_all_folds_average(id_ls, current_id)

            p1.do_run = False  # stop the thread
            p1.join()
