# -*- coding: utf-8 -*-
# @Time    : 7/5/21 5:23 PM
# @Author  : Jingnan
# @Email   : jiajingnan2222@gmail.com
import argparse
import datetime
import os
import shutil
import time
from typing import Union, Tuple
from medutils.medutils import icc
from mlflow.tracking import MlflowClient
import sys
import numpy as np
import nvidia_smi
import pandas as pd
import torch
from filelock import FileLock
from torch.utils.data import WeightedRandomSampler

from lung_function.modules.confusion_test import confusion
import threading
from pathlib import Path

import argparse
import datetime
import os
import shutil
import time
from typing import Union, Tuple
import threading

import numpy as np
import nvidia_smi
import pandas as pd
from filelock import FileLock
from pathlib import Path
from mlflow import log_metric, log_param, log_artifacts, log_params
import psutil
from mlflow import log_metric, log_param, start_run, end_run, log_params, log_artifact

def sampler_by_disext(tr_y, sys_ratio=None) -> WeightedRandomSampler:
    """Balanced sampler according to score distribution of disext.

    Args:
        tr_y: Training labels.
            - Three scores per image: [[score1_disext, score1_gg, score1_ret], [score2_disext, score2_gg, score3_ret],
             ...]
            - One score per image: [score1_disext, score2_disext, ...]
        sys_ratio:

    Returns:
        WeightedRandomSampler

    Examples:
        :func:`lung_function.modules.mydata.LoadScore.load`
    """
    disext_list = []
    for sample in tr_y:
        if type(sample) in [list, np.ndarray]:
            disext_list.append(sample[0])
        else:
            disext_list.append(sample)
    disext_np = np.array(disext_list)
    disext_unique = np.unique(disext_np)
    disext_unique_list = list(disext_unique)

    class_sample_count = np.array([len(np.where(disext_np == t)[0]) for t in disext_unique])
    if sys_ratio:
        weight = 1 / class_sample_count
        weight_sum = np.sum(weight)
        weight = np.array([w / weight_sum for w in weight])  # normalize the sum of weights to 1
        weight = (1 - sys_ratio) * weight  # scale the sume of weights to (1-sys_ratio)
        idx_0 = disext_unique_list.index(0)
        weight[idx_0] += sys_ratio
        sys_ratio_in_0 = sys_ratio / weight[idx_0]



        # weight[idx_0] += 20 * weight[idx_0]
        # samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])
        #
        # weight_0 = sys_ratio + (1-sys_ratio)/21  # weight for category of 0, which is for original 0 and sys 0
        # weight_others = 1 - weight_0  # weight for other categories
        # # weight = [weight_0, *weight_others]
        # samples_weight = np.array([weight_0 if t==0 else weight_others for t in disext_np])
        # print("weight: ", weight)
        # print(samples_weight)
    else:
        weight = 1. / class_sample_count

    print("class_sample_count", class_sample_count)
    print("unique_disext", disext_unique_list)
    print("weight: ", weight)

    samples_weight = np.array([weight[disext_unique_list.index(t)] for t in disext_np])

    # weight = [nb_nonzero/len(data_y_list) if e[0] == 0 else nb_zero/len(data_y_list) for e in data_y_list]
    samples_weight = samples_weight.astype(np.float32)
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    print(list(sampler))
    if sys_ratio:
        return sampler, sys_ratio_in_0
    else:
        return sampler


def get_mae_best(fpath: str) -> float:
    """Get minimum mae.

    Args:
        fpath: A csv file in which the `mae` at each epoch is recorded

    Returns:
        Minimum mae

    Examples:
        :func:`lung_function.modules.tool.eval_net_mae`

    """

    loss = pd.read_csv(fpath)
    mae = min(loss['mae'].to_list())
    return mae


def eval_net_mae(mypath, mypath2) -> float:
    """Copy trained model and loss log to new directory and get its valid_mae_best.

    Args:
        mypath: Current experiment Path instance
        mypath2: Trained experiment Path instance, if mypath is empty, copy files from mypath2 to mypath

    Returns:
        valid_mae_minimum

    Examples:
        :func:`lung_function.run.train` and :func:`lung_function.run_pos.train`

    """
    shutil.copy(mypath2.model_fpath, mypath.model_fpath)  # make sure there is at least one model there
    for mo in ['train', 'validaug', 'valid', 'test']:
        try:
            shutil.copy(mypath2.loss(mo), mypath.loss(mo))  # make sure there is at least one model
        except FileNotFoundError:
            pass
    valid_mae_best = get_mae_best(mypath2.loss('valid'))
    print(f'load model from {mypath2.model_fpath}, valid_mae_best is {valid_mae_best}')
    return valid_mae_best


def add_best_metrics(df: pd.DataFrame,
                     mypath,
                     mypath2,
                     index: int) -> pd.DataFrame:
    """Add best metrics: loss, mae (and mae_end5 if possible) to `df` in-place.

    Args:
        df: A DataFrame saving metrics (and other super-parameters)
        mypath: Current Path instance
        mypath2: Old Path instance, if the loss file can not be find in `mypath`, copy it from `mypath2`
        index: Which row the metrics should be writen in `df`

    Returns:
        `df`

    Examples:
        :func:`lung_function.modules.tool.record_2nd`

    """
    modes = ['train', 'validaug', 'valid', 'test']
    if mypath.project_name == 'score':
        metrics_min = 'mae_end5'
    else:
        metrics_min = 'mae'
    df.at[index, 'metrics_min'] = metrics_min

    for mode in modes:
        lock2 = FileLock(mypath.loss(mode) + ".lock")
        # when evaluating/inference old models, those files would be copied to new the folder
        with lock2:
            try:
                loss_df = pd.read_csv(mypath.loss(mode))
            except FileNotFoundError:  # copy loss files from old directory to here

                shutil.copy(mypath2.loss(mode), mypath.loss(mode))
                try:
                    loss_df = pd.read_csv(mypath.loss(mode))
                except FileNotFoundError:  # still cannot find the loss file in old directory, pass this mode
                    continue

            best_index = loss_df[metrics_min].idxmin()
            loss = loss_df['loss'][best_index]
            mae = loss_df['mae'][best_index]
            if mypath.project_name == 'score':
                mae_end5 = loss_df['mae_end5'][best_index]
                df.at[index, mode + '_mae_end5'] = round(mae_end5, 2)
        df.at[index, mode + '_loss'] = round(loss, 2)
        df.at[index, mode + '_mae'] = round(mae, 2)
    return df


def get_df_id(record_file: str) -> Tuple[pd.DataFrame, int]:
    """Get the current experiment ID. It equals to the latest experiment ID + 1.

    Args:
        record_file: A file to record experiments details (super-parameters and metrics).

    Returns:
        dataframe and new_id

    Examples:
        :func:`lung_function.modules.tool.record_1st`

    """
    if not os.path.isfile(record_file) or os.stat(record_file).st_size == 0:  # empty?
        new_id = 1
        df = pd.DataFrame()
    else:
        df = pd.read_csv(record_file)  # read the record file,
        last_id = df['ID'].to_list()[-1]  # find the last ID
        new_id = int(last_id) + 1
    return df, new_id


def record_1st(record_file) -> int:
    Path(record_file).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(record_file + ".lock")  # lock the file, avoid other processes write other things
    with lock:  # with this lock,  open a file for exclusive access
        with open(record_file, 'a'):
            df, new_id = get_df_id(record_file)
            idatime = {'ID': new_id}
            if len(df) == 0:  # empty file
                df = pd.DataFrame([idatime])  # need a [] , or need to assign the index for df
            else:
                index = df.index.to_list()[-1]  # last index
                for key, value in idatime.items():  # write new line
                    df.at[index + 1, key] = value  #

            df.to_csv(record_file, index=False)
            shutil.copy(record_file, record_file + "_cp")

    return new_id


def _bytes_to_megabytes(value_bytes: int) -> float:
    """Convert bytes to megabytes.

    Args:
        value_bytes: bytes number

    Returns:
        megabytes

    Examples:
        :func:`lung_function.modules.tool.record_gpu_info`

    """
    return round((value_bytes / 1024) / 1024, 2)


def record_mem_info() -> int:
    """

    Returns:
        Memory usage in kB

    .. warning::

        This function is not tested. Please double check its code before using it.

    """

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]
    print('int(memusage.strip())')

    return int(memusage.strip())


def record_gpu_info(outfile) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`lung_function.run.gpu_info` and :func:`lung_function.run_pos.gpu_info`

    """

    if outfile:
        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        # log_dict['gpuname'] = gpuname
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem_usage = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total)) + ' MB'
        # log_dict['gpu_mem_usage'] = gpu_mem_usage
        gpu_util = 0
        for i in range(5):
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            gpu_util += res.gpu
            time.sleep(1)
        gpu_util = gpu_util / 5
        # log_dict['gpu_util'] = str(gpu_util) + '%'
        return gpuname, gpu_mem_usage, str(gpu_util) + '%'
    else:
        print('outfile is None, can not show GPU memory info')
        return None, None, None

# def log_metrics_for_cgpu():
#     t0 = time.time()
#     size = q_step.qsize()
#     if size:
#         for i in range(size):
#             i = q_step.get()
#             log_metric('cpu_mem_used_GB_rss', q_cpu_mem_rss.get(), step=i)
#             # log_metric('cpu_mem_used_GB_in_process_vms', q_cpu_mem_vms.get(), step=i)
#             log_metric('cpu_util_used_percent', q_cpu_util_percent.get(), step=i)
#             # log_metric('cpu_mem_used_percent', q_cpu_mem_percent.get(), step=i)
#             log_metric("gpu_util", q_gpu_util.get(), step=i)
#             log_metric('gpu_mem_used_MB', q_gpu_mem_Mb.get(), step=i)
#
#         print(f'log_metrics_for_cgpu loged {size} steps, which cost {time.time() - t0} seconds.')
#

def record_cgpu_info(outfile, lock) -> Tuple:
    """Record GPU information to `outfile`.

    Args:
        outfile: The format of `outfile` is: slurm-[JOB_ID].out

    Returns:
        gpu_name, gpu_usage, gpu_util

    Examples:

        >>> record_gpu_info('slurm-98234.out')

        or

        :func:`ssc_scoring.run.gpu_info` and :func:`ssc_scoring.run_pos.gpu_info`

    """
    t = threading.currentThread()
    t.do_run = True

    if outfile:
        cpu_count = psutil.cpu_count()
        log_param('cpu_count', cpu_count)

        pid = os.getpid()
        python_process = psutil.Process(pid)

        jobid_gpuid = outfile.split('-')[-1]
        tmp_split = jobid_gpuid.split('_')[-1]
        if len(tmp_split) == 2:
            gpuid = tmp_split[-1]
        else:
            gpuid = 0
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuid)
        gpuname = nvidia_smi.nvmlDeviceGetName(handle)
        gpuname = gpuname.decode("utf-8")
        log_param('gpuname', gpuname)
        # log_dict['gpuname'] = gpuname

        # log_dict['gpu_mem_usage'] = gpu_mem_usage
        # gpu_util = 0
        i = 0
        period = 2  # 2 seconds
        # cgpu_dt = {'step': [],
        #             'cpu_mem_used_GB_in_process_rss': [],
        #            'cpu_mem_used_GB_in_process_vms': [],
        #            'cpu_util_used_percent': [],
        #            'cpu_mem_used_percent': [],
        #            'gpu_util': [],
        #            'gpu_mem_used_MB': [],
        #            }
        while i<60*20:  # stop signal passed from t, monitor 20 minutes
            if t.do_run:
                # q_step.put(i)

                memoryUse = python_process.memory_info().rss / 2. ** 30  # memory use in GB...I think
                # q_cpu_mem_rss.put(memoryUse)


                memoryUse2 = python_process.memory_info().vms / 2. ** 30  # memory use in GB...I think
                # q_cpu_mem_vms.put(memoryUse2)

                cpu_percent = psutil.cpu_percent()
                # q_cpu_util_percent.put(cpu_percent)
                # gpu_mem = dict(psutil.virtual_memory()._asdict())
                # log_params(gpu_mem)
                cpu_mem_used = psutil.virtual_memory().percent
                # q_cpu_mem_percent.put(cpu_mem_used)

                res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                # gpu_util += res.gpu
                # q_gpu_util.put(res.gpu)

                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                # gpu_mem_used = str(_bytes_to_megabytes(info.used)) + '/' + str(_bytes_to_megabytes(info.total))
                gpu_mem_used = _bytes_to_megabytes(info.used)
                # q_gpu_mem_Mb.put(gpu_mem_used)

                try:
                    with lock:
                        print('get lock by sub-thread')
                        log_metric('cpu_mem_used_GB_in_process_rss', memoryUse, step=i)
                        log_metric('cpu_mem_used_GB_in_process_vms', memoryUse2, step=i)
                        log_metric('cpu_util_used_percent', cpu_percent, step=i)
                        log_metric('cpu_mem_used_percent', cpu_mem_used, step=i)
                        log_metric("gpu_util", res.gpu, step=i)
                        log_metric('gpu_mem_used_MB', gpu_mem_used, step=i)
                        print('release lock by sub-thread')
                except Exception as er:  # sometimes the sqlite database is locked by the main thread.
                    print(er, file=sys.stderr)
                    pass
                time.sleep(period)
                i += period
            else:
                print('record_cgpu_info do_run is True, let stop the process')
                break
        print('It is time to stop this process: record_cgpu_info')
        return None
        # gpu_util = gpu_util / 5
        # gpu_mem_usage = str(gpu_mem_used) + ' MB'

        # log_dict['gpu_util'] = str(gpu_util) + '%'
        # return gpuname, gpu_mem_usage, str(gpu_util) + '%'


    else:
        print('outfile is None, can not show GPU memory info')
        return None


def record_artifacts(outfile):
    mythread = threading.currentThread()
    mythread.do_run = True
    if outfile:
        t = 0
        while 1:  # stop signal passed from t
            if mythread.do_run:
                log_artifact(outfile + '_err.txt')
                log_artifact(outfile + '_out.txt')
                if t <= 600:  # 10 minutes
                    period = 10
                    t += period
                else:
                    period = 60
                time.sleep(period)
            else:
                print('record_artifacts do_run is True, let stop the process')
                break

        print('It is time to stop this process: record_artifacts')
        return None
    else:
        print(f"No output file, no log artifacts")
        return None