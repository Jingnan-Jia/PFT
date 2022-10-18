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


def compare(dt1: dict, dt2: dict, dt_label: dict):
    """
    Compare each individual's difference of df1 and df2.
    1. select the common columns

    """
    try:
        assert len(dt1) == len(dt2)
    except Exception:
        print(len(dt1),len(dt2), len(dt_label))

    df1, df2, df_label = [dt['data'] for dt in [dt1, dt2, dt_label]]


    assert len(df1) == len(df2) == len(df_label) # check length is the same
    assert list(df1.index) == list(df2.index) == list(df_label.index)  # check the index (pat_id) is the same
    common_columns: list = list(df1.columns & df2.columns & df_label.columns) # a list of common column names
    if 'pat_id' in common_columns:
        common_columns.remove('pat_id')


    df1, df2, df_label = [df[common_columns] for df in [df1, df2, df_label]]

    data_name1 = f"Ex_{dt1['ex_id']}_{dt1['mode']}"
    data_name2 = f"Ex_{dt2['ex_id']}_{dt2['mode']}"

    row_nb = 1
    col_nb = len(common_columns)
    height_fig = 8 * row_nb
    length_fig = height_fig * col_nb
    fig_bin = plt.figure(figsize=(length_fig, height_fig))  # for bin plot, show the prediction distribution
    fig_sca = plt.figure(figsize=(length_fig, height_fig))  # for scatter plot, show the error distribution
    fig_stock = plt.figure(figsize=(length_fig, height_fig))  # for stock plot, show the error d
    fig_abs_stock = plt.figure(figsize=(length_fig, height_fig))  # for stock plot, show the error d


    for ax_idx, col_name in enumerate(common_columns):
        x = np.arange(len(df1))  # the label locations
        width = 0.35  # the width of the bars
        y1 = df1[col_name]
        y2 = df2[col_name]
        label = df_label[col_name]

        err1 = y1 - label
        err2 = y2 - label

        me1 = err1.mean()
        me2 = err2.mean()

        abs_err1 = (y1 - label).abs()
        abs_err2 = (y2 - label).abs()
        mae1 = abs_err1.mean()
        mae2 = abs_err2.mean()


        ax = fig_bin.add_subplot(row_nb, col_nb, ax_idx+1)
        rects1 = ax.bar(x - width/2, y1, width, label=data_name1, zorder=0 )
        rects2 = ax.bar(x + width/2, y2, width, label=data_name2, zorder=1)
        ax.scatter(x,label.to_numpy(), s=8, zorder=3)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Network output')
        ax.set_title(f"Comparison of network output on ({mode}) {col_name}")
        ax.legend()


        ax2 = fig_sca.add_subplot(row_nb, col_nb, ax_idx+1)
        rects1 = ax2.bar(x - width/2, err1, width, label=list(dt1.keys())[0], zorder=0 )
        rects2 = ax2.bar(x + width/2, err2, width, label=list(dt2.keys())[0] , zorder=1)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax2.text(0.5, 0.95, f"me1={me1:.2f}, me2={me2:.2f}", size=15, transform=ax2.transAxes, ha="center")
        ax2.text(0.5, 0.9, f"mae1={mae1:.2f}, mae2={mae2:.2f}", size=15, transform=ax2.transAxes, ha="center")

        ax2.set_ylabel('Network output error')
        ax2.set_title(f"Comparison of network output error on ({mode}) {col_name}")
        ax2.legend()


        ax3 = fig_stock.add_subplot(row_nb, col_nb, ax_idx+1)
        for i, e1, e2 in zip(x, err1, err2):
            if e2>e1: # error increases
                color = 'r'
            else:
                color = 'g'
            ax3.plot([i,i], [e1, e2], linewidth=2,color=color)

        ax3.plot([x[0],x[-1]], [0,0], color='k')
        ax3.text(0.5, 0.97, f"me1={me1:.2f}, me2={me2:.2f}", size=15, transform=ax3.transAxes, ha="center")
        ax3.text(0.5, 0.94, f"mae1={mae1:.2f}, mae2={mae2:.2f}", size=15, transform=ax3.transAxes, ha="center")

        # retrive icc and r values from mlflow
        icc1, r1 = dt1['run_dt'][f"icc_{mode}_{col_name}"], dt1['run_dt'][f"r_{mode}_{col_name}"]
        icc2, r2 = dt2['run_dt'][f"icc_{mode}_{col_name}"], dt2['run_dt'][f"r_{mode}_{col_name}"]
        ax3.text(0.5, 0.91, f"icc1={float(icc1):.2f}, icc2={float(icc2):.2f}", size=15, transform=ax3.transAxes, ha="center")
        ax3.text(0.5, 0.88, f"R1={float(r1):.2f}, R2={float(r2):.2f}", size=15, transform=ax3.transAxes, ha="center")
        ax3.set_ylabel('Network output error')
        ax3.set_title(f"Comparison of network output error ({mode}) on {col_name} [Red: Error increasement]")
        # ax3.legend()


        ax4 = fig_abs_stock.add_subplot(row_nb, col_nb, ax_idx+1)
        for i, e1, e2 in zip(x, abs_err1, abs_err2):
            if e2>e1: # error increases
                color = 'r'
            else:
                color = 'g'
            ax4.plot([i,i], [e1, e2], linewidth=2,color=color)

        ax4.plot([x[0],x[-1]], [0,0], color='k')

        ax4.text(0.5, 0.97, f"me1={me1:.2f}, me2={me2:.2f}", size=15, transform=ax4.transAxes, ha="center")
        ax4.text(0.5, 0.94, f"mae1={mae1:.2f}, mae2={mae2:.2f}", size=15, transform=ax4.transAxes, ha="center")
        # retrive icc and r values from mlflow
        ax4.text(0.5, 0.91, f"icc1={float(icc1):.2f}, icc2={float(icc2):.2f}", size=15, transform=ax4.transAxes, ha="center")
        ax4.text(0.5, 0.88, f"R1={float(r1):.2f}, R2={float(r2):.2f}", size=15, transform=ax4.transAxes, ha="center")
        ax4.set_ylabel('Network output absolute error')
        ax4.set_title(f"Comparison of network output absolute error ({mode}) on {col_name} [Red: Error increasement]")

        print(f"{col_name} successful!")

    fig_bin.tight_layout()
    fig_sca.tight_layout()
    fig_stock.tight_layout()
    fig_abs_stock.tight_layout()

    current_dir = pathlib.Path(__file__).resolve().parent
    fig_bin.savefig(current_dir / f"diff_bin_{mode}_{dt1['ex_id']}_{dt2['ex_id']}")
    fig_sca.savefig(current_dir / f"diff_error_bin_{mode}_{dt1['ex_id']}_{dt2['ex_id']}")
    fig_stock.savefig(current_dir / f"diff_stock_{mode}_{dt1['ex_id']}_{dt2['ex_id']}")
    fig_abs_stock.savefig(current_dir / f"diff_abs_stock_{mode}_{dt1['ex_id']}_{dt2['ex_id']}")

    # plt.show()




if __name__ == "__main__":
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1405_1404_1411_1410/16pats_pred.csv"
    # label_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

    basic_dir = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/"
    ex1 = '825'
    ex2 = '821'

    # retrive the icc and r values from mlflow
    mlflow.set_tracking_uri("http://nodelogin02:5000")

    experiment = mlflow.set_experiment("lung_fun_db15")
    client = MlflowClient()
    run_dt1, run_dt2 = [client.search_runs(experiment_ids=[experiment.experiment_id],
                                filter_string=f"params.id LIKE '%{ex}%'")[0].data.params for ex in [ex1, ex2]]

    for mode in ['train', 'valid', 'test']:
        print(f"start the mode: {mode} ==============")
        pred_fpath1 = basic_dir + f"{ex1}/{mode}_pred.csv"
        pred_fpath2 = basic_dir + f"{ex2}/{mode}_pred.csv"
        label_fpath = basic_dir + f"{ex1}/{mode}_label.csv"

        pred_df1, pred_df2, label_df = map(pd.read_csv, [pred_fpath1, pred_fpath2, label_fpath])
        dt1 = {'mode': mode, 'ex_id': ex1, "data": pred_df1, 'run_dt': run_dt1}
        dt2 = {'mode': mode, 'ex_id': ex2, "data": pred_df2, 'run_dt': run_dt2}

        dt_label = {'mode': mode, 'ex_id': ex1, 'data': label_df}

        compare(dt1, dt2, dt_label)
