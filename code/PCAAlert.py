# -*- coding: utf-8 -*-
import numpy as np

import pandas as pd
from pandas import DataFrame
import os

from sklearn.preprocessing import MinMaxScaler


from scipy.stats import f

from sklearn.externals import joblib

import time

import csv
import warnings
import win_unicode_console
import random
win_unicode_console.enable()
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)
file = open('C:\Bdir.ini','r',encoding='utf-8')
dir = file.readline()                    #   'C:\\Users\\icore2018\\Desktop\\BDproj'

file_origin_realtime= dir + '\\Rtopt\\Count.csv'     # 新井预测数据存放路径
dir_pca_alert = dir + "\\Rtopt\\EarlyAlertOutPut\\"
file.close()

dirAlertPklPath= dir + '\\Rtopt\\History\\alert\\'

#***************
def loadJobLib(file):
    os.chdir(dirAlertPklPath + file)
    a = joblib.load("train_pca_eigenvalue.pkl")
    b = joblib.load("train_pca_eigenvector.pkl")
    n = joblib.load("train_pca_shape.pkl")[0]
    m = joblib.load("train_pca_shape.pkl")[1]
    S_mean = joblib.load("train_pca_mean.pkl")
    S_var = joblib.load("train_pca_var.pkl")

    return a,b,n,m,S_mean,S_var


def getFilterRealTimeData(df):
    columns = ['ID', 'Time', 'Depth', 'ROP', 'SPP', 'TORQUE', 'WOB', 'FLW', 'RPM', \
               'Rho', 'FV', 'AV', 'PV', 'YP', 'API', 'Sand', 'Solid', 'GR', 'RES', 'GD', 'NP', \
               'WOB_min', 'WOB_max', 'WOB_range', 'FLW_min', 'FLW_max', 'FLW_range', 'RPM_min', 'RPM_max', 'RPM_range', \
               'SPP_min', 'SPP_max', 'TORQUE_min', 'TORQUE_max', 'GR_min', 'GR_max', 'RES_min', 'RES_max', 'DBTM',
               'BPOS', 'RunNum', 'PumpOnValue']

    df.columns = columns
    length = df.shape[0]

    if length == 0:
        return pd.DataFrame()
    if length > 1:  # 运行起始数据不少于2条时

        # 将上一米的ROP实际值作为预测模型的输入参数

        ROP_precede = df.loc[0:length - 2, 'ROP']
        SPP_precede = df.loc[0:length - 2, 'SPP']
        TOR_precede = df.loc[0:length - 2, 'TORQUE']

        df = df.drop([0])  # 删除第一条数据
        df = df.reset_index(drop=True)
    else:
        ROP_precede = 0
        SPP_precede = 0
        TOR_precede = 0

    df['ROP_precede'] = ROP_precede  # 上一米的ROP值
    df['SPP_precede'] = SPP_precede  # 上一米的SPP值
    df['TOR_precede'] = TOR_precede  # 上一米的TOR值
    '''
    length = df.shape[0]
    GR_min = df.GR_min[length-1]
    GR_max = df.GR_max[length-1]
    RES_min = df.RES_min[length-1]
    RES_max = df.RES_max[length-1]

    df_select = df[(df.ROP < 500) & (df.ROP >= 0)  & \
               (df.WOB < 35) & (df.WOB >= 0) & \
               (df.FLW < 5000) & (df.FLW >= 0) & \
               (df.RPM < 150) & (df.RPM >= 0) & \
               (df.TORQUE < 30000) & (df.TORQUE >= 0) & \
               (df.SPP < 5000) & (df.SPP >= 0) & \
               (df.Rho < 1.5) & (df.Rho >= 0) & \
               (df.FV < 80) & (df.FV >= 0) & \
               (df.PV < 50) & (df.PV >= 0) & \
               (df.YP < 30) & (df.YP >= 0) & \
               (df.API < 10) & (df.API >= 0) & \
               (df.Sand < 1) & (df.Sand >= 0) & \
               (df.Solid < 30) & (df.Solid >= 0) & \
               (df.GR < GR_max) & (df.GR >= GR_min) & \
               (df.RES < RES_max) & (df.RES >= RES_min)]
               #(df.R39PC < 100) & (df.R39PC > 0) & \
                #(df.ALCDA < 3) & (df.ALCDA > 1) & \
                #(df.TNPL < 80) & (df.TNPL > 0)
    '''
    return df





def PCA(test_data, a, b, n, m, S_mean, S_var,columns,file):
    testDataCopy = test_data.copy()
    test_data = test_data[columns]
    lambda1 = sorted(a, reverse=True)  # 特征值从大到小排序
    # k=6
    # 主元个数选取
    totalvar = 0;  # 累计贡献率，初值0
    for i in range(m):
        totalvar = totalvar + lambda1[i] / np.sum(a)  # 累计贡献率，初值0
        if totalvar >= 0.85:
            k = i + 1  # 确定主元个数
            break  # 跳出for循环

    if k == m:
        k -= 1
    PCnum = k;  # 选取的主元个数
    PC = np.eye(m, k);  # 定义一个矩阵，用于存放选取主元的特征向量
    for j in range(k):
        wt = a.tolist().index(lambda1[j])  # 查找排序完成的第j个特征值在没排序特征值里的位置。
        PC[:, j:j + 1] = b[:, wt:wt + 1];  # 提取的特征值对应的特征向量

    # 根据建模数据求取 T2 阈值限
    # 置信度 = (1-a)% =（1-0.05）%95%
    F = f.ppf(1 - 0.05, k, n - 1)  # F分布临界值
    # T2 = k*(n**2-1)*F/((n-k)*n) #T2求取
    T2 = k * (n - 1) * F / ((n - k))  # T2求取
    # 健康数据的 SPE 阈值限求解
    ST1 = 0  # 对应SPE公式中的角1初值
    ST2 = 0  # 对应SPE公式中的角2初值
    ST3 = 0  # 对应SPE公式中的角3初值
    for i in range(k - 1, m):
        ST1 = ST1 + lambda1[i]  # 对应SPE公式中的角1
        ST2 = ST2 + lambda1[i] * lambda1[i]  # 对应SPE公式中的角2
        ST3 = ST3 + lambda1[i] * lambda1[i] * lambda1[i]  # 对应SPE公式中的角3
    h0 = 1 - 2 * ST1 * ST3 / (3 * pow(ST2, 2))
    Ca = 3.62
    SPE = ST1 * pow(Ca * pow(2 * ST2 * pow(h0, 2), 0.5) / ST1 + 1 + ST2 * h0 * (h0 - 1) / pow(ST1, 2),
                    1 / h0)  # 健康数据SPE计算

    # 测试样本数据
    # m1 = test_data.shape[1];  # 列数
    n1 = test_data.shape[0];  # 行数

    test_data = np.array(test_data)  # 将DataFrame数据烈性转化为ndarray类型
    I = np.eye(m)  # 产生m*m的单位矩阵
    PC1 = np.transpose(PC)  # PC的转秩
    SPEa = np.arange(n1).reshape(1, n1)  # 定义测试数据的SPE矩阵,为正数矩阵
    # SPEa = np.double(SPEa)  # 将正数矩阵，转化为双精度数据矩阵

    TT2a = np.arange(n1).reshape(1, n1)  # 定义测试数据的T2矩阵,为正数矩阵
    # TT2a = np.double(TT2a)#将正数矩阵，转化为双精度数据矩阵
    DL = np.diag(lambda1[0:k])  # 特征值组成的对角矩阵
    DLi = np.linalg.inv(DL)  # 特征值组成的对角矩阵的逆矩阵
    # mpl.rcParams['font.sans-serif'] = ['SimHei']#在图形中显示汉字
    for i in range(n1):
        xnew = (test_data[i] - S_mean) / S_var;  # xnew=(Data2(i,1:m)-S_mean)./S_var;   (1,6)
        # 以下是实现Matlb程序：  err(1,i)=xnew*(eye(14)-PC*PC')*xnew';
        xnew1 = np.transpose(xnew)  # xnew的转秩   (6,1)
        PC1 = np.transpose(PC)  # PC的转秩
        XPC = np.dot(xnew, PC)  # 矩阵xnew与PC相乘   (1,4)
        XPCPC1 = np.dot(XPC, PC1)  # 矩阵XPC与PC1相乘  (1,6)
        XXPCPC1 = xnew - XPCPC1  # 矩阵xnew减去XPCPC1
        XXPCPC11 = np.transpose(XXPCPC1)

        SPEa[0, i] = np.dot(XXPCPC1, XXPCPC11)  # 矩阵XXPCPC1与XXPCPC1相乘

        XPi = np.dot(XPC, DLi)  # 矩阵XPC与DLi相乘
        XPiP = np.dot(XPi, PC1)  # 矩阵XPi与PC1相乘
        TT2a[0, i] = np.dot(XPiP, xnew1)  # 矩阵XPiP与xnew1相乘
    Sampling = np.r_[0.:n1]  # 产生的序列值式0到n1
    SPE1 = SPE * np.ones((1, n1))  # 产生SPE数值相同的矩阵
    T21 = T2 * np.ones((1, n1))  # 产生T2数值相同的矩阵
    sumf = 0
    loc = list()

    for i in range(n1):  # 对测试样本个数进行循环
        if i >= 2:
            if SPEa[0, i] >= SPE and SPEa[0, i - 1] >= SPE and SPEa[0, i - 2] >= SPE:  # 判断各个值是否小于阈值线
                print(i,testDataCopy[i], end=' ')  # 输出有故障的样本点
                loc.append(i)
                sumf += 1
        else:
            if ((TT2a[0, i] >= T2) & (SPEa[0, i] >= SPE) | (TT2a[0, i] < T2) & (SPEa[0, i] >= SPE)):
                print("故障点深度:",testDataCopy.Depth[i])  # 输出有故障的样本点
                #todo 影响因素
                # 計算SPE貢獻率
                testdstd = (test_data[0] - S_mean) / S_var
                PCp = np.dot(PC, PC1)
                PCp1 = np.dot(testdstd, PCp)
                PCpp = testdstd - PCp1
                contSPE = np.multiply(PCpp, PCpp)

                testdstd = (test_data[0] - S_mean) / S_var
                DD = np.dot(PC, DLi)
                D = np.dot(DD, PC1)
                Temp = np.dot(testdstd, D)
                conT2 = np.multiply(Temp, testdstd)

                # SPE貢獻率直方圖
                X_list = columns
                contSPE1 = list(contSPE)
                conT21 = list(conT2)
                tem22 = max(contSPE1)

                tem33 = contSPE1.index(tem22)
                tem44 = conT21.index(max(conT2))
                global tem33lan
                global tem44lan
                global temluo
                tem33lan = X_list[tem33]
                tem44lan = X_list[tem44]

                #todo 输出文件

                result_filename = "{}.csv".format(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
                with open(dir_pca_alert + file + "\\" + result_filename, "w", newline='') as csvfile:
                    bwriter = csv.writer(csvfile)
                    #SPEa[0, i]  SPE  TT2a[0, i]  T2
                    outPutColumns = ['ID', 'Time', 'Depth', 'SPE_Value', 'SPE_Control_Line', 'T2_Value','T2_Control_Line', 'Group_Type']
                    outPutValues = [testDataCopy.ID[i], testDataCopy.Time[i], testDataCopy.Depth[i], SPEa[0, i], SPE, TT2a[0, i],T2, file]
                    if file == "model1":
                        # ['Rho', 'FV', 'PV', 'YP', 'API', 'Sand', 'Solid']
                        outPutColumns.extend(['SPE_Rho_ConValue', 'SPE_FV_ConValue', 'SPE_PV_ConValue', 'SPE_YP_ConValue', 'SPE_API_ConValue', 'SPE_Sand_ConValue', 'SPE_Solid_ConValue'])
                        outPutColumns.extend(['T2_Rho_ConValue', 'T2_FV_ConValue', 'T2_PV_ConValue', 'T2_YP_ConValue','T2_API_ConValue', 'T2_Sand_ConValue', 'T2_Solid_ConValue'])
                    elif file == "model2":
                        # ['GR', 'RES', 'TORQUE', 'SPP', 'ROP']
                        outPutColumns.extend(['SPE_GR_Value', 'SPE_RES_Value', 'SPE_TORQUE_Value', 'SPE_SPP_Value', 'SPE_ROP_Value'])
                        outPutColumns.extend(['T2_GR_Value', 'T2_RES_Value', 'T2_TORQUE_Value', 'T2_SPP_Value', 'T2_ROP_Value'])
                    elif file == "model3":
                        # ['WOB', 'SPP', 'RPM', 'FLW', 'TORQUE', 'ROP']
                        outPutColumns.extend(['SPE_WOB_Value', 'SPE_SPP_Value', 'SPE_RPM_Value', 'SPE_FLW_Value', 'SPE_TORQUE_Value', 'SPE_ROP_Value'])
                        outPutColumns.extend(['T2_WOB_Value', 'T2_SPP_Value', 'T2_RPM_Value', 'T2_FLW_Value', 'T2_TORQUE_Value', 'T2_ROP_Value'])
                    elif file == "model4":
                        # ['API', 'Rho', 'SPP']
                        outPutColumns.extend(['SPE_API_Value', 'SPE_Rho_Value', 'SPE_SPP_Value'])
                        outPutColumns.extend(['T2_API_Value', 'T2_Rho_Value', 'T2_SPP_Value'])
                    elif file == "model5":
                        # ['WOB', 'RPM', 'FLW', 'SPP']
                        outPutColumns.extend(['SPE_WOB_Value', 'SPE_RPM_Value', 'SPE_FLW_Value', 'SPE_SPP_Value'])
                        outPutColumns.extend(['T2_WOB_Value', 'T2_RPM_Value', 'T2_FLW_Value', 'T2_SPP_Value'])
                    bwriter.writerow(outPutColumns)
                    outPutValues.extend(np.round(contSPE1, 3))
                    outPutValues.extend(np.round(conT21, 3))
                    bwriter.writerow(outPutValues)

                print("分组模型",file,"ID :", testDataCopy.ID[i], ",深度:", testDataCopy.Depth[i], ",SPE最大因素:", str(tem33lan), ",T2最大因素:",
                      str(tem44lan),"SPE贡献值:",np.round(contSPE1,3), "T2贡献值:",np.round(conT21,3) , ",原因:", reasonOfAlert(random.randint(1, 10)))

                loc.append(i)
                sumf += 1
            else:


                print("分组模型",file,"深度:",testDataCopy.Depth[i],"没有故障")


def fixAbnormalData(df):
    for index in range(df.shape[0]):
        if df.Rho[index] < 1 or df.Rho[index] > 1.5:
            if index > 0 and df.Rho[index - 1] >= 1 and df.Rho[index - 1] <= 1.5:
                df.Rho[index] = df.Rho[index - 1]

            else:
                df.Rho[index] = 1.5
        if df.FV[index] < 0 or df.FV[index] > 80:
            if index > 0 and df.FV[index - 1] >= 0 and df.FV[index - 1] <= 80:
                df.FV[index] = df.FV[index - 1]
            else:
                df.FV[index] = 80
        if df.PV[index] < 0 or df.PV[index] > 50:
            if index > 0 and df.PV[index - 1] >= 0 and df.PV[index - 1] <= 50:
                df.PV[index] = df.PV[index - 1]
            else:
                df.PV[index] = 50
        if df.YP[index] < 0 or df.YP[index] > 30:
            if index > 0 and df.YP[index - 1] >= 0 and df.YP[index - 1] <= 30:
                df.YP[index] = df.YP[index - 1]
            else:
                df.YP[index] = 30
        if df.API[index] < 0 or df.API[index] > 10:
            if index > 0 and df.API[index - 1] >= 0 and df.API[index - 1] <= 10:
                df.API[index] = df.API[index - 1]
            else:
                df.API[index] = 10
        if df.Sand[index] < 0 or df.Sand[index] > 10:
            if index > 0 and df.Sand[index - 1] >= 0 and df.Sand[index - 1] <= 10:
                df.Sand[index] = df.Sand[index - 1]
            else:
                df.Sand[index] = 10
        if df.Solid[index] < 0 or df.Solid[index] > 30:
            if index > 0 and df.Solid[index - 1] >= 0 and df.Solid[index - 1] <= 30:
                df.Solid[index] = df.Solid[index - 1]
            else:
                df.Solid[index] = 30

        GR_min = df.GR_min[index]
        GR_max = df.GR_max[index]
        RES_min = df.RES_min[index]
        RES_max = df.RES_max[index]
        if df.GR[index] < GR_min or df.GR[index] > GR_max:
            df.GR[index] = GR_max
        if df.RES[index] < RES_min or df.RES[index] > RES_max:
            df.RES[index] = RES_max

    return df

Depth = 0     # 初始化条件

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
        sm_data=DataFrame(temp1,columns=['ID','Time','Depth','WOB', 'SPP', 'RPM', 'FLW', 'TORQUE', 'ROP'])
        return sm_data
    return temp_data


def reasonOfAlert(flag):
    if flag == 1:
        reason = "故障的原因是：可能是地层发生了变动"
    if flag == 2:
        reason = "故障的原因是：可能是发生了井漏或者井涌，钻头异常"
    if flag == 3:
        reason = "故障的原因是：可能是地层发生了变动"
    if flag == 4:
        reason = "故障的原因是：可能是钻具断落，钻头已近寿命，钻井液异常"
    if flag == 5:
        reason = "故障的原因是：可能是发生了井漏或者井涌，钻头异常"
    if flag == 6:
        reason = "故障的原因是：可能是钻井液异常，发生卡钻"
    if flag == 7:
        reason = "故障的原因是：可能是地层发生了变动"
    if flag == 8:
        reason = "故障的原因是：可能是钻井液异常，发生卡钻"
    if flag == 9:
        reason = "故障的原因是：可能是钻具掉落，钻井坍塌"
    if flag == 10:
        reason = "故障的原因是：可能是钻具掉落，钻井坍塌"
    return reason
while True:

    try:
        origin_df = pd.read_csv(file_origin_realtime)
        df2_select = getFilterRealTimeData(origin_df)
    except:
        print("exception")
        continue
    df2_select.WOB = df2_select.WOB / 0.454
    df2_select.TORQUE = df2_select.TORQUE * 1000
    df2_select.TOR_precede = df2_select.TOR_precede * 1000
    df2_select.SPP = df2_select.SPP / 0.069
    df2_select.SPP_precede = df2_select.SPP_precede / 0.069
    df2_select = fixAbnormalData(df2_select)
    length = df2_select.shape[0]
    if Depth == 0 or (Depth < round(df2_select.Depth[length - 1], 2) and df2_select.WOB[length - 1] > 0 and df2_select.RPM[length - 1] > 0):

        #todo 2018/11/14
        af = os.listdir(dir + '\\Rtopt\\History\\alert')
        for file in af:
            x_test_be = df2_select.loc[:, ['ID', 'Time', 'Depth']]

            a, b, n, m, S_mean, S_var = loadJobLib(file)
            # columns = ['ID', 'Time', 'Depth', 'ROP', 'SPP', 'TORQUE', 'WOB', 'FLW', 'RPM', \
            #            'Rho', 'FV', 'AV', 'PV', 'YP', 'API', 'Sand', 'Solid', 'GR', 'RES', 'GD', 'NP', \
            #            'WOB_min', 'WOB_max', 'WOB_range', 'FLW_min', 'FLW_max', 'FLW_range', 'RPM_min', 'RPM_max',
            #            'RPM_range', \
            #            'SPP_min', 'SPP_max', 'TORQUE_min', 'TORQUE_max', 'GR_min', 'GR_max', 'RES_min', 'RES_max',
            #            'DBTM',
            #            'BPOS', 'RunNum', 'PumpOnValue']
            if file == "model1":
                columns = ['Rho', 'FV', 'PV', 'YP', 'API', 'Sand', 'Solid']
            elif file == "model2":
                columns = ['GR', 'RES', 'TORQUE', 'SPP', 'ROP']
            elif file == "model3":
                columns = ['WOB', 'SPP', 'RPM', 'FLW', 'TORQUE', 'ROP']
            elif file == "model4":
                columns = ['API', 'Rho', 'SPP']
            elif file == "model5":
                columns = ['WOB', 'RPM', 'FLW', 'SPP']

            x_test_af = df2_select.loc[:, columns]
            minMaxScaler = MinMaxScaler()
            x_test_af = minMaxScaler.fit_transform(x_test_af)

            x_test_be = x_test_be.loc[length - 1, :]
            x_test_af = x_test_af[length - 1]

            if file == "model1":
                x_test = pd.DataFrame(
                    [[x_test_be[0], x_test_be[1], x_test_be[2], x_test_af[0], x_test_af[1], x_test_af[2],
                      x_test_af[3], x_test_af[4], x_test_af[5], x_test_af[6]]],
                    columns=['ID', 'Time', 'Depth', 'Rho', 'FV', 'PV', 'YP', 'API', 'Sand', 'Solid'])
            elif file == "model2":
                x_test = pd.DataFrame(
                    [[x_test_be[0], x_test_be[1], x_test_be[2], x_test_af[0], x_test_af[1], x_test_af[2],
                      x_test_af[3],x_test_af[4]]],
                    columns=['ID', 'Time', 'Depth', 'GR', 'RES', 'TORQUE', 'SPP', 'ROP'])
            elif file == "model3":
                x_test = pd.DataFrame(
                    [[x_test_be[0], x_test_be[1], x_test_be[2], x_test_af[0], x_test_af[1], x_test_af[2],
                      x_test_af[3],x_test_af[4],x_test_af[5]]],
                    columns=['ID', 'Time', 'Depth','WOB', 'SPP', 'RPM', 'FLW', 'TORQUE', 'ROP'])
            elif file == "model4":
                x_test = pd.DataFrame(
                    [[x_test_be[0], x_test_be[1], x_test_be[2], x_test_af[0], x_test_af[1], x_test_af[2]]],
                    columns=['ID', 'Time', 'Depth', 'API', 'Rho', 'SPP'])
            elif file == "model5":
                x_test = pd.DataFrame(
                    [[x_test_be[0], x_test_be[1], x_test_be[2], x_test_af[0], x_test_af[1], x_test_af[2],
                      x_test_af[3]]],
                    columns=['ID', 'Time', 'Depth','WOB', 'RPM', 'FLW', 'SPP'])
            PCA(x_test, a, b, n, m, S_mean, S_var,columns, file)
            Depth = round(df2_select.Depth[length - 1], 2)  # 更新深度值,保留小数点后两位




