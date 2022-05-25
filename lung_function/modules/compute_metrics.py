import sys

sys.path.append("../..")
import csv
import glob
import os
import threading
from typing import List

import SimpleITK as sitk
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import ndimage
import glob
import os
import seaborn as sns
# sns.set_theme(color_codes=True)

import matplotlib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score

import lung_function.modules.my_bland as sm

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def icc(label_fpath, pred_fpath):
    icc_dict = {}

    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    if 'ID' == label.columns[0]:
        del label["ID"]
    if 'ID' == pred.columns[0]:
        del pred["ID"]

    original_columns = label.columns

    # ori_columns = list(label.columns)

    label['ID'] = np.arange(1, len(label) + 1)
    label['rater'] = 'label'

    pred['ID'] = np.arange(1, len(pred) + 1)
    pred['rater'] = 'pred'

    data = pd.concat([label, pred], axis=0)

    for column in original_columns:
        icc = pg.intraclass_corr(data=data, targets='ID', raters='rater', ratings=column).round(2)
        icc = icc.set_index("Type")
        icc = icc.loc['ICC2']['ICC']
        prefix = label_fpath.split("/")[-1].split("_")[0]
        icc_dict['icc_' + prefix + '_' + column] = icc

    return icc_dict

def metrics(pred_fpath, label_fpath):
    df_pred = pd.read_csv(pred_fpath)
    df_label = pd.read_csv(label_fpath)
    print('len_df_label', len(df_label))


    lower_y_ls, upper_y_ls = [], []
    lower_x_ls, upper_x_ls = [], []

    fig = plt.figure(figsize=(15, 4))
    fig_2 = plt.figure(figsize=(15, 5))
    fig_3 = plt.figure(figsize=(15, 5))


    if len(df_label.columns) == 3:
        row_nb, col_nb = 1, 3
    else:
        raise Exception(f'the columns number is not 3, it is {len(df_label.columns)} ', df_label.columns)

    basename = os.path.dirname(pred_fpath)
    prefix = pred_fpath.split("/")[-1].split("_")[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']

    for plot_id, column in enumerate(df_label.columns):
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)

        # bland-altman plot
        ax = fig.add_subplot(row_nb, col_nb, plot_id + 1)
        ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)

        # f, ax = plt.subplots(1, figsize=(8, 5))
        scatter_kwds = {'c': colors[plot_id], 'label': column}


        f = sm.mean_diff_plot(pred, label, ax=ax, scatter_kwds=scatter_kwds,
                              bland_in_1_mean_std=None,
                              adap_markersize=False)
        f_2 = sm.mean_diff_plot(pred, label, ax=ax_2, sd_limit=0, scatter_kwds=scatter_kwds,
                                bland_in_1_mean_std=None,
                                adap_markersize=False, ynotdiff=True)

        ax.set_title(column, fontsize=15)
        ax_2.set_title(column, fontsize=15)


        lower_y, upper_y = ax.get_ybound()  # set these plots as the same scale for comparison
        lower_x, upper_x = ax.get_xbound()
        lower_y_ls.append(lower_y)
        upper_y_ls.append(upper_y)
        lower_x_ls.append(lower_x)
        upper_x_ls.append(upper_x)

        diff = pred.astype(int) - label.astype(int)
        abs_diff = np.abs(diff)
        ave_mae = np.mean(abs_diff)
        std_mae = np.std(abs_diff)
        mean = np.mean(diff)
        std = np.std(diff)

        print(f"ave_mae for {column} is {ave_mae}")
        print(f"std_mae for {column} is {std_mae}")
        print(f"mean for {column} is {mean}")
        print(f"std for {column} is {std}")
        print("Finish plot of ", column)

    for plot_id, column in enumerate(df_label.columns):
        label = df_label[column].to_numpy().reshape(-1, )
        pred = df_pred[column].to_numpy().reshape(-1, )
        ax_3 = fig_3.add_subplot(row_nb, col_nb, plot_id + 1)
        ax_3 = sns.regplot(x=label, y=pred, color=colors[plot_id])


    for plot_id, column in enumerate(df_label.columns):
        label = df_label[column].to_numpy().reshape(-1, )
        pred = df_pred[column].to_numpy().reshape(-1, )

        ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)
        # plot linear regression line
        m, b = np.polyfit(label, pred, 1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label, pred)
        x_reference = np.array([0, 256])
        print(column, 'linear regression m, b:', m, b)
        print(column, 'linear regression m, b, r^2:', slope, intercept, r_value ** 2)

        ax_2.plot(x_reference, m * x_reference + b, '--', color='gray')  # light gray
        # ax_2.text(0.1, 0.7, '---  Regression line',
        #           ha="left", fontsize='large', transform=ax_2.transAxes)
        ax_2.text(0.1, 0.7, f'y = {m:.2f}x + {b:.2f}\nR\N{SUPERSCRIPT TWO} = {r_value ** 2: .2f}',
                  ha="left", fontsize='large', transform=ax_2.transAxes)
    print(f"lower_y_ls: {lower_y_ls}, upper_y_ls: {upper_y_ls}")
    lower_y, upper_y = min(lower_y_ls), max(upper_y_ls)
    lower_x, upper_x = min(lower_x_ls), max(upper_x_ls)

    print("lower:", lower_y, "upper:", upper_y)
    common_y = max(abs(lower_y), abs(upper_y))
    common_x = max(abs(lower_x), abs(upper_x))

    for i in range(row_nb * col_nb):
        if df_label.columns[i] == 'DLCO_SB':
            limitx = 15  # max value of FVC
        else:
            limitx = 7  # max value of FEV1 and DLCO_SB

        ax = fig.add_subplot(row_nb, col_nb, i + 1)
        ax.set_xlim(0, limitx)
        # ax.set_ylim(-common_y * 1.2, common_y * 1.2)

        ax_2 = fig_2.add_subplot(row_nb, col_nb, i + 1)
        ax_2.set_xlim(0, limitx)
        ax_2.set_ylim(0, limitx)

        ax_3 = fig_3.add_subplot(row_nb, col_nb, i + 1)
        ax_3.set_xlim(0, limitx)
        ax_3.set_ylim(0, limitx)

    # f.suptitle(prefix.capitalize() + " Bland-Altman Plot", fontsize=26)
    f.tight_layout()
    f.savefig(basename + '/' + prefix + '_bland_altman.png')
    plt.close(f)

    # f_2.suptitle(prefix.capitalize() + " Prediction Scatter Plot", fontsize=26)
    f_2.tight_layout()
    f_2.savefig(basename + '/' + prefix + '_scatter.png')
    plt.close(f_2)

    fig_3.tight_layout()
    fig_3.savefig(basename + '/' + prefix + '_scatter_ci.png')
    plt.close(fig_3)

if __name__ == "__main__":
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1405_1404_1411_1410/16pats_pred.csv"
    # label_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

    pred_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_pred.csv"
    label_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_label.csv"

    metrics(pred_fpath, label_fpath)
    icc_value = icc(label_fpath, pred_fpath)
    print('icc:', icc_value)