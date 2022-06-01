
import sys
from typing import List

sys.path.append("../..")
from mlflow import log_metric, log_param, log_params
import mlflow
import threading
import time
from medutils.medutils import count_parameters
from medutils import medutils
from queue import Queue
import torch
import torch.nn as nn
import sqlite3
import copy
from lung_function.modules.datasets import all_loaders
from lung_function.modules.loss import get_loss
from lung_function.modules.networks import get_net_3d
from lung_function.modules.path import PFTPath
from lung_function.modules.set_args import get_args
from lung_function.modules.tool import record_1st, record_artifacts, record_cgpu_info
from lung_function.modules.compute_metrics import icc, metrics
import random
args = get_args()
global_lock = threading.Lock()

def sub_thread():
    for i in range(500):  ## 100 seconds
        # with global_lock:
        tmp2 = random.random()
        log_metric('Accuracy2', tmp2, step=i)
        print(f'accuracy2: {tmp2}')
        # time.sleep(0.5)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://nodelogin02:5000")
    with mlflow.start_run():
        p1 = threading.Thread(target=sub_thread)
        p1.start()

        p2 = threading.Thread(target=record_cgpu_info, args=(args.outfile, global_lock))
        p2.start()

        for i in range(500): ## 100 seconds
            # with global_lock:
            tmp1 = random.random()
            log_metric('Accuracy1', tmp1, step=i)
            print(f'accuracy1: {tmp1}')
            time.sleep(0.5)


        p1.join()
        p2.join()

    print("Finished!")