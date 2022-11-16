"""
Inspired by Irene's idea, this script aims to show the difference of each individual's prediction.

Total workflow:
1. load the predictions of two experiments.
2. Compare the predictions of different targets (categories), see if the prediction increase or not.
3. Calculate the error change, see if the error increase or not.

"""


import sys

sys.path.append("../../../..")
import csv
import glob
import os
import threading
from typing import List
import pathlib
import SimpleITK as sitk
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import ndimage
import glob
import os
import seaborn as sns
# sns.set_theme(color_codes=True)
from lung_function.modules.compute_metrics import icc, metrics

import matplotlib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score
import mlflow
from mlflow.tracking import MlflowClient

import lung_function.modules.my_bland as sm

# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt





if __name__ == "__main__":
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1405_1404_1411_1410/16pats_pred.csv"
    # label_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

    basic_dir = "/data1/jjia/lung_function/lung_function/scripts/results/experiments"

    ex_ls = [i+1 for i in [872, 881, 877, 879, 902, 904, 1001, 1004, 1015, 1017, 1019, 1021, 1024]]

    # retrive the icc and r values from mlflow
    mlflow.set_tracking_uri("http://nodelogin02:5000")

    experiment = mlflow.set_experiment("lung_fun_db15")
    client = MlflowClient()
    run_ls = [client.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.id LIKE '%{ex}%'")[0].data.params for ex in ex_ls]



    for mode in ['valid']:
        print(f"start the mode: {mode} ==============")
        label_fpath = f"{basic_dir}/{ex_ls[0]}/{mode}_label.csv"
        pred_path_dt = {k:  f"{basic_dir}/{k}/{mode}_pred.csv" for k in ex_ls}

        label = pd.read_csv(label_fpath)
        pred_dt = {k: pd.read_csv(pred_fpth) for k, pred_fpth in pred_path_dt.items()}

        err_dt = {k: pred - label for k, pred in pred_dt.items()}

        err = [v for k, v in err_dt.items()]
        # concatenate them
        # err_sum = pd.DataFrame()
        # for i in err:
        #     err_sum += i
        # err_sum = err_sum/len(err)
        columns = [i for i in label.columns if i!='pat_id']
        data_ls = [[] for i in columns]
        for idx, v in enumerate(columns):
            for er in err:
                data_ls[idx].append(er[v])

        
        for column, data in zip(columns, data_ls):
            data = np.array(data)
            fig = plt.figure(figsize=(8, 4)) 
            ax = fig.add_subplot(1, 1, 1)


            # # Creating axes instance
            # ax = fig.add_axes([0, 0, 1, 1])
            
            # Creating plot
            bp = ax.boxplot(data)
            ax.plot([0, 64], [0,0])
            ax.set_xlabel(label['pat_id'].to_list())
            ax.set_title(f"Error of {column} in {len(ex_ls)} experiments (different nets/pretraining)")

            if column == 'TLC_He':
                labelss=label['pat_id'].to_list()
                ax.set_xticks(list(range(len(labelss))))
                ax.set_xticklabels(labelss, rotation=90, ha='right')
            plt.show()
            plt.savefig(f"{column}.jpg")



        print('finish')

