import xlrd as xd
import numpy as np
from scipy.stats import kendalltau
import pandas as pd


def save_to_excel(matrix, file_name='E:\\paper\\code\\SLH_EN\\results\\P1.xlsx'):
    df = pd.DataFrame(matrix)
    df.to_excel(file_name, index=False, header=False)
    print(f'SIR_matrix has been saved to {file_name}')


def sckendall_1(a, b):
    kendall_tau, p_value = kendalltau(a, b)  # kendalltau：系统自带肯德尔系数
    # kendall_tau = sckendall(a, b)#sckendall：自己定义的肯德尔系数，好像
    # print(kendall_tau_2)
    # print(kendall_tau)
    return kendall_tau



def sckendall_test(pred_value, name):

    pred_value = pred_value.to('cpu').tolist()
    flattened_list = [item[0] for item in pred_value] #从pred_value列表中提取值

    '肯德尔系数'
    H_1 = []
    H_1.extend([flattened_list])  #
    # A)不同网络a取值不同，需要换成对应的
    print(H_1)


    data = xd.open_workbook('./dataset/data/data_set_1.xls')  # 打开excel表所在路径
    sheet = data.sheet_by_name(name)  # 读取数据，以excel表名来打开
    SIR = []
    for r in range(sheet.ncols):  # 将表中数据按列逐步添加到列表中，最后转换为list结构
        data1 = []
        for c in range(sheet.nrows):
            data1.append(sheet.cell_value(c, r))
        SIR.append(list(data1))
    # print("SIR", SIR)
    # print(len(SIR))

    SIR_matrix = np.zeros([len(H_1), len(SIR)])
    for i in range(len(H_1)):
        for j in range(len(SIR)):
            SIR_matrix[i][j] = sckendall_1(H_1[i], SIR[j])
    # 保存 SIR_matrix 到 Excel
    save_to_excel(SIR_matrix)

    return SIR_matrix






