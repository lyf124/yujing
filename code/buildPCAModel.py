
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
import os
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from scipy.stats import f
import numbers
import tkinter.filedialog
from sklearn.externals import joblib
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import csv


def select_well_data(lst, name):
    x = DataFrame()
    y = DataFrame()
    for data in lst:
        m = data[select_names][:-1]
        # m = data[[name]][:-1]
        n = data[[name]][1:]
        x = x.append(m, ignore_index=True)
        y = y.append(n, ignore_index=True)
    # 可以对x, y进行特征工程，模糊处理之类的, 归一化， 标准化
    # 变动x就行，y不用变

    # x = x.apply(lambda x: (x - np.mean(x))/(np.std(x)))

    #
    # x = x.drop(name, axis=1)
    y.rename(columns={name: name + '+1'}, inplace=True)
    data_all = pd.concat([x, y], axis=1)
    # 将数据前17口井为训练集， 后1口井为测试集
    train_len = sum(lst[i].shape[0] - 1 for i in range(len(file_list) - 1))
    x_train, x_test = x[0:train_len], x[train_len:]
    y_train, y_test = y[0:train_len], y[train_len:]
    # 将数据根据test_size大小随机分开，得到训练集，测试集
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)
    # 打印x_train, y_train, x_test, y_test的shape
    return x_train, x_test, y_train, y_test, name, data_all, x, y

#數據平滑處理
def Data_smoothing(data,pcatrain_names):
    temp_data=data[pcatrain_names]
    tn1=temp_data.shape[0]
    if tn1>=3:
        tx=np.array(temp_data)
        temp1=np.transpose(tx)
        tempx=[]
        for i in range(len(pcatrain_names)):
            tempx.append([])
        for i in range(len(pcatrain_names)):
            tempx[i].append(temp1[i][0])

            tempx[i].append(temp1[i][1])
            for tc in range(2,tn1):
                tempp=0.33*temp1[i][tc]+0.33*temp1[i][tc-1]+0.34*temp1[i][tc-2]
                tempx[i].append(tempp)

        temp1=np.transpose(tempx)
        sm_data=DataFrame(temp1)
        return sm_data


def PCAz(x_train,x_test,ver):
    if ver == 1:
        pcatrain_names = ['Rho','FV','PV','YP','API','sand','solid']               ###1组
    if ver == 2:
        pcatrain_names = ['DGRCC','R27PC', 'TORQUElbft' , 'SPPpsi', 'ROPmhr']   ###2组
    if ver == 3:
        pcatrain_names = ['WOBklbs','SPPpsi','RPM', 'FLWpmpslmn','TORQUElbft','ROPmhr']         ###3 组
    if ver == 4:
        pcatrain_names = ['API','Rho','SPPpsi']          ###4组
    if ver == 5:
        pcatrain_names = ['WOBklbs','RPM','FLWpmpslmn','SPPpsi'] ###5组

    # if ver ==5:
    #     pcatrain_names = ['API','ALCDLC']               ###5组
    # if ver ==2:
    #     pcatrain_names = ['ALCDLC','API','Rho','SPPpsi']   ###2组
    # if ver ==4:
    #     pcatrain_names = ['WOBklbs','SPPpsi','RPM', 'FLWpmpslmn','TORQUElbft','ROPmhr']         ###4 组（强加）
    # if ver ==1:
    #     pcatrain_names = ['Rho','API','solid','sand']          ###1组
    # if ver ==6:
    #     pcatrain_names = ['FV','PV','YP'] ###6组
    # if ver ==3:
    #     pcatrain_names = ['TNPL','ALCDLC']                  ######3组
        #pcatrain_names = ['WOBklbs','SPPpsi','RPM', 'FLWpmpslmn','TORQUElbft','ROPmhr']
        #pcatrain_names = ['ROPmhr','FLWpmpslmn','Rho','API','TNPL'] ###1，3,7,11
    train_data=x_train[pcatrain_names]
    test_data=x_test[pcatrain_names]
    #數據平滑處理
    #train_data = Data_smoothing(x_train,pcatrain_names)
    test_data  = Data_smoothing(x_test,pcatrain_names)
    return PCA(train_data,test_data,len(pcatrain_names),ver)


def PCA(train_data, test_data, long, ver):
    m = train_data.shape[1];  # 列数
    n = train_data.shape[0];  # 行数
    # 数据标准化处理
    S_mean = np.mean(train_data)  # 健康数据矩阵的列均值
    S_mean = np.array(S_mean)
    S_var = np.std(train_data, ddof=1)
    S_var = np.array(S_var)
    train_data -= S_mean
    train_data /= S_var
    train_data = np.where(train_data < 4.0e+11, train_data, 0.0)
    X_new = train_data;
    X_new = np.transpose(X_new);
    Z = np.dot(X_new, train_data / (n - 1))
    # Z = np.dot(X_new,train_data/(n-1))
    a, b = np.linalg.eig(Z)
    joblib.dump(a, '../params/train_pca_eigenvalue.pkl')
    joblib.dump(b, '../params/train_pca_eigenvector.pkl')
    joblib.dump(train_data.shape, '../params/train_pca_shape.pkl')
    joblib.dump(S_mean, '../params/train_pca_mean.pkl')
    joblib.dump(S_var,'../params/train_pca_var.pkl')
    mean = joblib.load('../params/train_pca_mean.pkl')
    shape = joblib.load('../params/train_pca_shape.pkl')


file_name = '../data85/'
# file_list = os.listdir('./data/')
file_list = ['final6.csv', 'final15.csv', 'final28.csv', 'final37.csv', 'final43.csv', 'final44.csv', 'final46.csv',
             'final47.csv','final56.csv','final59.csv','final63.csv','final112.csv','final342.csv']

lst = [pd.read_csv(file_name + i) for i in file_list]

df = DataFrame()
for i in lst:
    df = df.append(i, ignore_index=True)  # df是整个数据总和
df.head()  # 查看数据前5行

select_names = ['TotDepth', 'ROPmhr', 'RPM', 'FLWpmpslmn', 'TORQUElbft', 'SPPpsi', 'WOBklbs', 'Rho', 'FV', \
                'PV', 'YP', 'API', 'sand', 'solid', 'DGRCC', 'R27PC', 'TNPL']
name = 'WOBklbs'
x_train1, x_test1, y_train, y_test, name, data_all, x, y = select_well_data(lst, name)

PCAz(x_train1,x_test1,5)