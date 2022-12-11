
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
from lung_function.modules.tool import (record_1st, record_cgpu_info, retrive_run)



mlflow.set_tracking_uri("http://nodelogin02:5000")
experiment = mlflow.set_experiment("lung_fun_db2")
record_fpath = "results/record.log"
id = record_1st(record_fpath)  # write super parameters from set_args.py to record file.

client = MlflowClient()
run_ls = client.search_runs(experiment_ids=[experiment.experiment_id], run_view_type=3,
                            filter_string=f"params.id LIKE '%{1269}%'")


print('yes')

print('no')