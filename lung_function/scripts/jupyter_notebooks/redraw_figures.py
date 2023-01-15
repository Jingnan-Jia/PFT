
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
from lung_function.modules.trans import batch_bbox2_3D
import sys
sys.path.append("../modules/networks/models_pcd")

args = get_args()

args.id = '2154'
args.ct_sp = '1.5'
modes = ['train', 'valid', 'test']

mypath = PFTPath(args.id, check_id_dir=False, space=args.ct_sp)
label_ls = [mypath.save_label_fpath(mode) for mode in modes]
pred_ls = [mypath.save_pred_fpath(mode) for mode in modes]

for pred_fpath, label_fpath in zip(pred_ls, label_ls):
    r_p_value = metrics(pred_fpath, label_fpath, ignore_1st_column=True, xy_same_max=True)
    log_params(r_p_value)
    print('r_p_value:', r_p_value)

    icc_value = icc(label_fpath, pred_fpath, ignore_1st_column=True)
    log_params(icc_value)
    print('icc:', icc_value)

print('Finish all things!')