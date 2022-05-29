# -*- coding: utf-8 -*-
# @Time    : 4/5/22 12:25 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
# log_dict is used to record super parameters and metrics

import sys
from typing import List

sys.path.append("../..")
from mlflow import log_metric, log_param, log_params
import mlflow
import threading
import time
from medutils.medutils import count_parameters
from medutils import medutils

import torch
import torch.nn as nn
import sqlite3

from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info
from lung_function.modules.compute_metrics import icc, metrics
args = get_args()

class Run:
    def __init__(self, args):
        self.mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
        self.device = torch.device("cuda")
        self.target = [i.lstrip() for i in args.target.split('-')]
        self.net = get_net_3d(name=args.net, nb_cls=len(self.target)) # output FVC and FEV1
        print('net:', self.net)

        net_parameters = count_parameters(self.net)
        net_parameters = str(round(net_parameters / 1000 / 1000, 2))
        log_param('net_parameters_M', net_parameters)

        self.data_dt = all_loaders(self.mypath.data_dir, self.mypath.label_fpath, args)

        self.loss_fun = get_loss(args.loss)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.net = self.net.to(self.device)

        if args.pretrained_id:
            pretrained_path = PFTPath(args.pretrained_id, check_id_dir=False, space=args.ct_sp)
            self.net.load_state_dict(torch.load(pretrained_path.model_fpath, map_location=self.device))  # model_fpath need to exist

        self.BestMetricDt = {'trainLossEpochBest': 1000,
                             'trainnoaugLossEpochBest': 1000,
                        'validLossEpochBest': 1000,
                        'testLossEpochBest': 1000,

                        'trainMAEEpoch_AllBest': 1000,
                             'trainnoaugMAEEpoch_AllBest': 1000,
                        'validMAEEpoch_AllBest': 1000,
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
                    head = self.target
                    batch_y_np = batch_y.cpu().detach().numpy()
                    pred_np = pred.cpu().detach().numpy()
                    medutils.appendrows_to(self.mypath.save_label_fpath(mode), batch_y_np, head=head)
                    medutils.appendrows_to(self.mypath.save_pred_fpath(mode), pred_np, head=head)

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
                torch.save(self.net.state_dict(), self.mypath.model_fpath)



def run(args):
    myrun = Run(args)
    infer_modes = ['train', 'trainnoaug', 'valid', 'test']
    if args.mode == 'infer':
        save_pred = True
        for mode in infer_modes:
            myrun.step(mode,  0,  save_pred)
    else: # 'train' or 'continue train'
        for i in range(args.epochs):  # 20000 epochs
            myrun.step('train', i)
            if i % args.valid_period == 0:  # run the validation
                myrun.step('valid',  i)
                myrun.step('test',  i)
            if i == args.epochs - 1:  # load best model and do inference
                print('start inference')
                myrun.net.load_state_dict(torch.load(myrun.mypath.model_fpath, map_location=myrun.device))  # model_fpath need to exist
                print(f"load net from {myrun.mypath.model_fpath}")

                save_pred = True
                for mode in infer_modes:
                    myrun.step(mode, i, save_pred)

    mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
    modes = ['train', 'trainnoaug', 'valid', 'test']
    label_ls = [mypath.save_label_fpath(mode) for mode in modes]
    pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

    for pred_fpath, label_fpath in zip(pred_ls, label_ls):
        metrics(pred_fpath, label_fpath)
        icc_value = icc(label_fpath, pred_fpath)
        log_params(icc_value)
        print('icc:', icc_value)

    print('Finish all things!')


if __name__ == "__main__":
    # mlflow.set_tracking_uri("http://10.161.27.235:5000")
    mlflow.set_tracking_uri("http://nodelogin02:5000")

    # mlflow.set_tracking_uri('sqlite:///mlrunsdb15.db')  # bread down at some time
    # mlflow.set_tracking_uri('http://localhost:5000')


    mlflow.set_experiment("lung_fun_db15")
    id = record_1st("results/record.log")  # write super parameters from set_args.py to record file.

    with mlflow.start_run(run_name=str(id), tags={"mlflow.note.content": args.remark}):
        # p1 = threading.Thread(target=record_cgpu_info, args=(args.outfile,))
        # p1.start()
        # p2 = threading.Thread(target=record_artifacts, args=(args.outfile,))
        # p2.start()

        args.id = id  # do not need to pass id seperately to the latter function
        log_params(vars(args))
        run(args)

        # p1.do_run = False  # stop the thread
        # p2.do_run = False  # stop the thread
        # p1.join()
        # p2.join()

