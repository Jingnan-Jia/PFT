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
import scipy.stats as stats
import statsmodels
from statsmodels import api
import matplotlib
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import cohen_kappa_score

import lung_function.modules.my_bland as sm

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt




def icc(label_fpath, pred_fpath, ignore_1st_column=False):
    icc_dict = {}

    label = pd.read_csv(label_fpath)
    pred = pd.read_csv(pred_fpath)
    if ignore_1st_column:
        pred = pred.iloc[: , 1:]
        label = label.iloc[: , 1:]
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

def metrics(pred_fpath, label_fpath, ignore_1st_column=False, xy_same_max=True):
    """
    ignore_1st_column: ignore the index column
    """
    
    r_dict, p_dict = {}, {}
    df_pred = pd.read_csv(pred_fpath)
    df_label = pd.read_csv(label_fpath)
    

    
    if ignore_1st_column:
        df_pred = df_pred.iloc[: , 1:]
        df_label = df_label.iloc[: , 1:]
    print('len_df_label', len(df_label))


    lower_y_ls, upper_y_ls = [], []
    lower_x_ls, upper_x_ls = [], []

    row_nb = 1
    col_nb = len(df_label.columns)
    height_fig = 5
    length_fig = height_fig * col_nb
    # if col_nb < 10:
    fig = plt.figure(figsize=(length_fig, height_fig))  # for bland-altman plot
    fig_2 = plt.figure(figsize=(length_fig, height_fig))  # for scatter plot
    # fig_3 = plt.figure(figsize=(length_fig, height_fig))  # for scatter plot with 95% CI
    # else:
    #     raise Exception(f"the columns number is greater than 10: {df_label.columns}")

    basename = os.path.dirname(pred_fpath)
    prefix = pred_fpath.split("/")[-1].split("_")[0]
    if  col_nb <= 11:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#808000']
    elif col_nb <= 19:
        colors = ['#7f0000', '#808000', '#3cb371', '#7f007f', '#008080', '#7f007f', '#ff0000', '#ff8c00', '#ffd700',
                  '#0000cd',
                  '#00ff7f', '#00ffff', '#adff2f', '#00bfff', '#ff00ff', '#f0e68c', '#dda0dd', '#ff1493', '#ffa07a',
                  ]
    else:
        raise Exception(f"the columns number is greater than 20: {df_label.columns}")

    # 将第一列放到第三列，保持其他列不变，把FVC调整到第三列去
    
    # print(df_label.columns[0]) 
    # if df_label.columns[0]!='pat_id':
    #     cols_names.insert(2, cols_names.pop(0))
    # else:
    #     cols_names.insert(3, cols_names.pop(1))
    df_pred = df_pred.rename(columns={'DLCOc_SB': 'DLCOc', 'TLC_He': 'TLC'})  
    df_label = df_label.rename(columns={'DLCOc_SB': 'DLCOc', 'TLC_He': 'TLC'})  
    print(df_pred.columns.tolist(),'yyaayy')
    # cols_names = df_pred.columns.tolist()
    # df_pred = df_pred[cols_names]
    # df_label = df_label[cols_names]  
    
    for plot_id, column in enumerate(df_label.columns):
        if column=='pat_id':
            continue
        label = df_label[column].to_numpy().reshape(-1, 1)
        pred = df_pred[column].to_numpy().reshape(-1, 1)

        # bland-altman plot
        ax = fig.add_subplot(row_nb, col_nb, plot_id + 1)
        ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)
        # ax_3 = fig_3.add_subplot(row_nb, col_nb, plot_id + 1)
        ax_2 = sns.regplot(x=label, y=pred, color=colors[plot_id], label = f"95% confidence interval")
        
        # predicting interval
        # import scipy.stats as stats
        # n = len(pred)                        # number of samples
        # m = 2                             # number of parameters
        # dof = n - m                       # degrees of freedom
        # t = stats.t.ppf(0.975, dof)       # Students statistic of interval confidence
        # std_error = (np.sum(residual**2) / dof)**.5   # Standard deviation of the error
        # residual = pred - label

        # # to plot the adjusted model
        # x_line = np.linspace(np.min(label), np.max(label), 100)
        # y_line = np.polyval([slope, intercept], x_line)

        # pi = t * std_error * (1 + 1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))**.5  
        # ax_2.fill_between(x_line, y_line + pi, y_line - pi, color = 'gray', alpha=.1, label = '95% prediction interval')

        
        # model = api.OLS(endog = pred, exog = label)
        # results = model.fit()
        # df = pd.DataFrame({'label': label.flatten(), 'pred': pred.flatten()})

        # predictions = results.get_prediction(df)
        # predictions.summary_frame(alpha=0.05)
        
        # ax_2.fill_between(df['temperature'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], alpha=.1, label='Prediction interval')
        # ax_2.fill_between(df['temperature'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')

        #Get two-tailed t-values:
        #Significance level:
        # sl = 0.05
        # (t_minus, t_plus) = stats.t.interval(alpha = (1.0 - sl), df =  len(results.resid) - len(x0) )
        # y_value_at_x0 = np.dot(results.params, x0)
        # lower_bound = y_value_at_x0 + t_minus*np.sqrt(results.mse_resid*( np.dot(np.dot(x0.T,results.normalized_cov_params),x0) ))
        # upper_bound = y_value_at_x0 +  t_plus*np.sqrt(results.mse_resid*( np.dot(np.dot(x0.T,results.normalized_cov_params),x0) ))

        if False:
            # from tsmoothie.smoother import *
            # # generate intervals
            # # operate smoothing
            # smoother = PolynomialSmoother(degree=1)
            # smoother.smooth(data)
            # low_pi, up_pi = smoother.get_intervals('prediction_interval', confidence=0.05)
            # plt.fill_between(range(len(smoother.data[0])), low_pi[0], up_pi[0], alpha=0.3, color='blue')
            
            # 添加常数项到 x
            X = api.add_constant(pred)
            # 执行线性回归
            model = api.OLS(label, X)
            results = model.fit()
            # 计算预测区间
            x_line = np.linspace(np.min(label), np.max(label), 100)
            X_new = api.add_constant(x_line)
            dt = results.get_prediction(X_new).summary_frame(alpha = 0.05)
            y_prd = dt['mean']
            yprd_ci_lower = dt['obs_ci_lower']
            yprd_ci_upper = dt['obs_ci_upper']
            
            # predicted_intervals = predictions.conf_int(alpha=0.05)  # 95% 预测区间
            # 绘制预测区间
            _ = ax_2.plot(x_line, yprd_ci_lower, color = "blue", linestyle = "--")
            _ = ax_2.plot(x_line, yprd_ci_upper, color = "blue", linestyle = "--")


            # ax_2.fill_between(X_new[:, 1], predicted_intervals[:, 0], predicted_intervals[:, 1], color='r', alpha=0.1, label='95% Prediction Interval')
                


            # # Prediction Interval
            # slope, intercept = np.polyfit(label.flatten(), pred.flatten(), 1)  # linear model adjustment

            # x_line = np.linspace(np.min(label), np.max(label), 100)
            # y_line = np.polyval([slope, intercept], x_line)

            # n = label.shape[0]
            # m = 2                             # number of parameters
            # dof = n - m                       # degrees of freedom
            # t = stats.t.ppf(0.95, dof)       # Students statistic of interval confidence
            # # slope, intercept = np.polyfit(label.flatten(), pred.flatten(), 1)  # linear model adjustment
            # # y_model = slope * x_line + intercept   # modeling...
            # residual = label - y_line
            # res_mean = np.mean(residual)
            # std_error = (np.sum((residual - res_mean )**2) / dof)**.5   # Standard deviation of the error
            # pi = t * std_error * np.sqrt(1 + 1/n + (x_line  - np.mean(label))**2 / np.sum((label - np.mean(label))**2))  
            # ax_2.fill_between(x_line, y_line - pi, y_line + pi, color = 'lightcyan', label = '95% prediction interval')

            # # ax_2.fill_between(x_line, y_model + pi, y_model - pi, color="None", linestyle="--")
            # ax_2.plot(x_line, y_model - pi, "--", color="0.5")
            # ax_2.plot(x_line, y_model + pi, "--", color="0.5")


        # f, ax = plt.subplots(1, figsize=(8, 5))
        scatter_kwds = {'c': colors[plot_id], 'label': column}


        f = sm.mean_diff_plot(pred, label, ax=ax, 
        scatter_kwds=scatter_kwds,
                              bland_in_1_mean_std=None,
                              adap_markersize=False, 
                              xy_same_max=xy_same_max)
        f_2 = sm.mean_diff_plot(pred, label, ax=ax_2, sd_limit=0, 
        scatter_kwds=scatter_kwds,
                                bland_in_1_mean_std=None,
                                adap_markersize=False, 
                                ynotdiff=True, 
                                xy_same_max=xy_same_max)

        ax.set_title(column, fontsize=15)
        ax_2.set_title(column, fontsize=15)

        
        # plot linear regression line
        m, b = np.polyfit(label.flatten(), pred.flatten(), 1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label.flatten(), pred.flatten())
        r_dict['r_' + prefix + '_' + column] = r_value
        p_dict['p_of_r' + prefix + '_' + column] = p_value

        x_reference = np.array([np.min(label), np.max(label)])
        print(column, 'linear regression m, b:', m, b)
        print(column, 'linear regression m, b, r^2:', slope, intercept, r_value ** 2)

        # ax_2.plot(x_reference, m * x_reference + b, '--', color='gray')  # light gray
        # ax_2.text(0.1, 0.7, '---  Regression line',
        #           ha="left", fontsize='large', transform=ax_2.transAxes)
        # ax_2.text(0.1, 0.7, f'y = {m:.2f}x + {b:.2f}\nR = {r_value: .2f}\nR\N{SUPERSCRIPT TWO} = {r_value ** 2: .2f}',
        #           ha="left", fontsize='large', transform=ax_2.transAxes)


        if p_value < 0.001:
            p_txt = 'p < 0.001'
        elif p_value < 0.01:
            p_txt = 'p < 0.01'
        elif p_value < 0.05:
            p_txt = 'p < 0.05'
        else:
            p_txt = f'p = {p_value:.2f}'
            
        ax_2.text(0.1, 0.8, f'R = {r_value: .2f}\n{p_txt}',
                  ha="left", fontsize='large', transform=ax_2.transAxes)

        min_xy = min(np.min(label), np.min(pred))
        max_xy = max(np.max(label), np.max(pred))
        
        ax_2.plot([0, max_xy*1.2], [0, max_xy*1.2], '--', color = 'gray')
        
        ax_2.set_xlim(0, max_xy*1.2)
        ax_2.set_ylim(0, max_xy*1.2)
        
        if column in ['DLCOc']:
            mre = 0.1
        elif column=='FEV1':
            mre = 0.06
        elif column=='FVC':
            mre = 0.05
        elif column=='TLC':
            mre = 0.1
        min_lb = np.min(label)
        max_lb = np.max(label)
        ax_2.plot([min_lb, max_lb], [min_lb * (1-mre), max_lb * (1-mre)], '--', color = 'gray')
        ax_2.plot([min_lb, max_lb], [min_lb * (1+mre), max_lb * (1+mre)], '--', color = 'gray')


        lower_y, upper_y = ax.get_ybound()  # set these plots as the same scale for comparison
        lower_x, upper_x = ax.get_xbound()
        lower_y_ls.append(lower_y)
        upper_y_ls.append(upper_y)
        lower_x_ls.append(lower_x)
        upper_x_ls.append(upper_x)

        diff = pred - label
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

    # for plot_id, column in enumerate(df_label.columns):
    #     label = df_label[column].to_numpy().reshape(-1, )
    #     pred = df_pred[column].to_numpy().reshape(-1, )
    #     ax_3 = fig_3.add_subplot(row_nb, col_nb, plot_id + 1)
    #     ax_3 = sns.regplot(x=label, y=pred, color=colors[plot_id])


    # for plot_id, column in enumerate(df_label.columns):
    #     label = df_label[column].to_numpy().reshape(-1, )
    #     pred = df_pred[column].to_numpy().reshape(-1, )

    #     ax_2 = fig_2.add_subplot(row_nb, col_nb, plot_id + 1)

    #     # plot linear regression line
    #     m, b = np.polyfit(label, pred, 1)
    #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(label, pred)
    #     r_dict['r_' + prefix + '_' + column] = r_value
    #     p_dict['p_of_r' + prefix + '_' + column] = p_value

    #     x_reference = np.array([np.min(label), np.max(label)])
    #     print(column, 'linear regression m, b:', m, b)
    #     print(column, 'linear regression m, b, r^2:', slope, intercept, r_value ** 2)

    #     ax_2.plot(x_reference, m * x_reference + b, '--', color='gray')  # light gray
    #     # ax_2.text(0.1, 0.7, '---  Regression line',
    #     #           ha="left", fontsize='large', transform=ax_2.transAxes)
    #     ax_2.text(0.1, 0.7, f'y = {m:.2f}x + {b:.2f}\nR = {r_value: .2f}\nR\N{SUPERSCRIPT TWO} = {r_value ** 2: .2f}',
    #               ha="left", fontsize='large', transform=ax_2.transAxes)
    # print(f"lower_y_ls: {lower_y_ls}, upper_y_ls: {upper_y_ls}")
    # lower_y, upper_y = min(lower_y_ls), max(upper_y_ls)
    # lower_x, upper_x = min(lower_x_ls), max(upper_x_ls)

    # print("lower:", lower_y, "upper:", upper_y)
    # common_y = max(abs(lower_y), abs(upper_y))
    # common_x = max(abs(lower_x), abs(upper_x))

    # for i in range(row_nb * col_nb):
    #     if df_label.columns[i] == 'DLCO_SB':
    #         limitx = 15  # max value of FVC
    #     elif df_label.columns[i] == 'TLC_He':
    #         limitx = 12
    #     else:
    #         limitx = 7  # max value of FEV1 and DLCO_SB
    #     limitx = None

    #     ax = fig.add_subplot(row_nb, col_nb, i + 1)
    #     # ax.set_xlim(0, limitx)
    #     # # ax.set_ylim(-common_y * 1.2, common_y * 1.2)

    #     ax_2 = fig_2.add_subplot(row_nb, col_nb, i + 1)
        # ax_2.set_xlim(0, limitx)
    #     # ax_2.set_ylim(0, limitx)

    #     ax_3 = fig_3.add_subplot(row_nb, col_nb, i + 1)
    #     # ax_3.set_xlim(0, limitx)
    #     # ax_3.set_ylim(0, limitx)

    # f.suptitle(prefix.capitalize() + " Bland-Altman Plot", fontsize=26)
    f.tight_layout()
    f.savefig(basename + '/' + prefix + '_bland_altman.png')
    plt.close(f)

    # f_2.suptitle(prefix.capitalize() + " Prediction Scatter Plot", fontsize=26)
    f_2.tight_layout()
    f_2.savefig(basename + '/' + prefix + '_scatter.png')
    plt.close(f_2)

    # fig_3.tight_layout()
    # fig_3.savefig(basename + '/' + prefix + '_scatter_ci.png')
    # plt.close(fig_3)

    all_dt = {**r_dict, **p_dict}
    return all_dt

if __name__ == "__main__":
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/LKT2_16patients.csv"
    # pred_fpath = "/data/jjia/ssc_scoring/ssc_scoring/results/models/1405_1404_1411_1410/16pats_pred.csv"
    # label_fpath = "/data/jjia/ssc_scoring/ssc_scoring/dataset/observer_agreement/16_patients/ground_truth_16patients.csv"

    pred_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_pred.csv"
    label_fpath = "/data1/jjia/lung_function/lung_function/scripts/results/experiments/202/traininfernoaug_label.csv"

    metrics(pred_fpath, label_fpath)
    icc_value = icc(label_fpath, pred_fpath)
    print('icc:', icc_value)