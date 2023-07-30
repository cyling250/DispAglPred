import os

import torch
import xlrd3 as xlrd
import numpy as np


def read_earthquake_wave(filename):
    # 单文件地震波数据
    PW = []
    PW_flag = False
    SW = []
    SW_flag = False
    with open(filename, "r") as fp:
        while True:
            i = fp.readline()  # 读取每一行
            if not i:
                break
            i = i.strip()  # 去除前后空格
            try:
                i = float(i)
            except Exception as ex:
                if i[0] == 'D':
                    # 这是地震波的步长
                    step = float(i[2:])
                    # print(float(step))
                    continue
                elif i == "PW:":
                    # 开始记录P波
                    PW_flag = True
                    SW_flag = False
                    continue
                elif i == "SW:":
                    # 开始记录S波
                    PW_flag = False
                    SW_flag = True
                    continue
                else:
                    print("\r", filename, ex, end="")
                    continue
            if PW_flag:
                PW.append(i)
            elif SW_flag:
                SW.append(i)
    print()
    return PW, SW, step


def read_file_x(folder_name):
    # 读取所有的地震波
    file_list = os.listdir(folder_name)
    PW = []
    SW = []
    step = []
    for i in range(len(file_list)):
        if os.path.splitext(file_list[i])[-1] == ".txt":
            file_name = folder_name + "/" + file_list[i]
            P, S, s = read_earthquake_wave(file_name)
            step.append(s)
            PW.append(P)
            SW.append(S)
    return PW, SW, step


def read_shear_force_result(file_name):
    # 读取基底剪力时程，x表示x方向的时程，y表示y方向的时程，两个都是一个二维列表
    # 第一维表示不同的地震波，第二位表示地震波的时程
    print("正在打开文件。。。")
    workbook = xlrd.open_workbook(file_name)
    print("正在读取表单。。。")
    sheet = workbook.sheet_by_index(0)
    x = []
    y = []
    step = []
    for i in range(100):
        temp_x = []
        temp_y = []
        print("正在读取第{}条地震波的基底剪力时程数据。。。".format(i))
        for j in range(1, sheet.nrows):
            if sheet.cell(j, i * 3).value == "":
                break
            temp_x.append(float(sheet.cell(j, i * 3 + 1).value))
            temp_y.append(float(sheet.cell(j, i * 3 + 2).value))
        x.append(temp_x)
        y.append(temp_y)
        step.append(float(sheet.cell(2, i * 3).value))
    print()
    return x, y, step


def read_displacement_result(file_name):
    # 读取顶层位移时程，x表示x方向的时程，y表示y方向的时程，两个都是一个二维列表
    # 第一维表示不同的地震波，第二位表示地震波的时程
    print("正在打开文件。。。")
    workbook = xlrd.open_workbook(file_name)
    print("正在读取表单")
    sheet = workbook.sheet_by_index(1)
    x = []
    y = []
    step = []
    for i in range(100):
        temp_x = []
        temp_y = []
        print("\r正在读取第{}条地震波的位移时程数据。。。".format(i), end="")
        for j in range(1, sheet.nrows):
            if sheet.cell(j, i * 3).value == "":
                break
            temp_x.append(float(sheet.cell(j, i * 3 + 1).value))
            temp_y.append(float(sheet.cell(j, i * 3 + 2).value))
        x.append(temp_x)
        y.append(temp_y)
        step.append(float(sheet.cell(2, i * 3).value))
    print()
    return x, y, step


def freq_conversion(res_list, res_freq, des_freq):
    '''
    :param reslist:初始序列
    :param resfreq: 初始序列频率
    :param desfreq: 目标序列频率
    :return: 目标序列
    '''
    # 使用内插方法对时间序列进行频率转换
    des_length = int((len(res_list) - 1) * res_freq / des_freq)
    if len(res_list) * res_freq % des_freq == 0:
        des_length += 1
    des_list = []
    for i in range(des_length):
        this_time = des_freq * i
        if this_time % res_freq == 0:
            des_list.append(res_list[int(this_time / res_freq)])
        else:
            interpolation = (res_list[int(this_time / res_freq) + 1] - res_list[int(this_time / res_freq)]) / res_freq
            interpolation = interpolation * (this_time - int(this_time / res_freq) * res_freq)
            interpolation = interpolation + res_list[int(this_time / res_freq)]
            des_list.append(interpolation)
    return des_list


def dim1_to_dim2(x):
    # 将一维时程转化为二维时程图
    for i in range(len(x)):
        temp = np.zeros((20, 100))
        for j in range(20):
            for k in range(100):
                temp[j][k] = x[i][j * 20 + k]
        x[i] = temp
    return x


def MinMaxScaler(x, PGA):
    for i in range(len(x)):
        if abs(max(x[i])) > abs(min(x[i])):
            quot = abs(PGA / max(x[i]))
        else:
            quot = abs(PGA / min(x[i]))
        for j in range(len(x[i])):
            x[i][j] *= quot
    return x


def read_cjwyj(file_name):
    # 读取层间位移角数据
    print("正在打开文件。。。")
    workbook = xlrd.open_workbook(file_name)
    print("正在读取表单")
    sheet = workbook.sheet_by_index(2)
    x = []
    y = []
    for i in range(100):
        temp_x = []
        temp_y = []
        print("\r正在读取第{}条地震波的层间位移角数据。。。".format(i), end="")
        for j in range(21):
            temp_x.append(float(sheet.cell(i * 2 + 1, j + 1).value))
            temp_y.append(float(sheet.cell(i * 2 + 2, j + 1).value))
        x.append(temp_x)
        y.append(temp_y)
    print()
    return x, y


def data_enhance(data):
    data_new = []
    for i in range(len(data)):
        temp = []
        for j in range(0, 2000 - 20 * 20, 2):
            temp1 = data[i][j:j + 20 * 20]
            temp1 = temp1.reshape(20, 20)
            temp.append(temp1.numpy())
        data_new.append(np.array(temp))
    return torch.Tensor(data_new)
