import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def create_dataset(dataset, slice_number):
    dataX = []
    for i in range(0, len(dataset), slice_number):
        a = dataset[i:(i + slice_number)]
        dataX.append(a)
    return np.array(dataX)


# ########################双线性插值#####dataset是3D
def interpolation(dataset, Height, Wide):
    dataX = []
    for i in range(0, len(dataset), 1):
        a = dataset[i, :, :]
        Height_a = len(a[:, 0])  # 21
        Wide_a = len(a[0, :])  # 21
        b = np.zeros((Height, Wide))
        for j in range(0, Height, 1):
            for k in range(0, Wide, 1):
                m = j * ((Height_a - 1) / (Height - 1))
                n = k * ((Wide_a - 1) / (Wide - 1))
                m1 = int(m)
                n1 = int(n)
                rH = m - m1
                rW = n - n1
                if k == Wide - 1 and j == Height - 1:
                    b[j, k] = a[m1, n1]
                elif j == Height - 1:
                    b[j, k] = (1 - rW) * a[m1, n1] + rW * a[m1, n1 + 1]
                elif k == Wide - 1:
                    b[j, k] = (1 - rH) * a[m1, n1] + rH * a[m1 + 1, n1]
                else:
                    b[j, k] = (1 - rW) * (1 - rH) * a[m1, n1] + (1 - rW) * rH * a[m1 + 1, n1] + rW * (1 - rH) * \
                              a[m1, n1 + 1] + rW * rH * a[m1 + 1, n1 + 1]
        dataX.append(b)
    return np.array(dataX)


def data2Ndata(dataset, n):
    dataX = []
    for i in range(0, n):
        dataX.append(dataset)
    return np.array(dataX)


# ########################削去右边线、底边线
def xuebian_dataset(dataset):
    dataX = []
    for i in range(0, dataset.shape[0]):
        a = dataset[i, 0:(dataset.shape[1] - 1), 0:(dataset.shape[2] - 1)]
        dataX.append(a)
    return np.array(dataX)


def mean_relative_error(y_true, y_pred):
    relative_error = 1 - abs(1 - y_pred / y_true)
    relative_error = relative_error.cpu()
    relative_error = relative_error.detach().numpy()
    relative_error = np.average(relative_error)
    return relative_error


def read_split_data(path_v10: str, path_u10: str, path_topo: str, path_swh: str, train_rate: float = 0.8):
    Height_wind = 41  # wind的单位矩阵纵向行数或切片距离
    Wide_wind = 9  # wind的单位矩阵横向列数
    Height_wind_chazhi = 41  # wind_插值的单位矩阵纵向行数
    Wide_wind_chazhi = 41  # wind_插值的单位矩阵横向列数
    Height_swh = 41  # swh的单位矩阵纵向行数或切片距离
    Wide_swh = 41  # swh的单位矩阵横向列数
    num_disorder = 1  # 随机种子数

    # 完全打乱
    data_u10 = pd.read_csv(path_u10, header=None)
    # data_u10_reduced = 0.1*data_u10  # 数据处理为0.1倍
    data_u10_reduced = data_u10
    data_u10_all = create_dataset(data_u10_reduced, Height_wind)  # 切割原数组为n多个9×9的
    # data_u10_all_2 = interpolation(data_u10_all, Height_wind_chazhi, Wide_wind_chazhi)  # 双线性插值为41×41的
    data_u10_all_3 = xuebian_dataset(data_u10_all)  # 削边线，去掉最右侧和最下侧
    data_u10_all_1 = np.expand_dims(data_u10_all_3, 3)
    ################################################################################
    data_u10_all_1_1 = data_u10_all_1[0:data_u10_all_1.shape[0] - 5, :, :, :]
    data_u10_all_1_2 = data_u10_all_1[1:data_u10_all_1.shape[0] - 4, :, :, :]
    data_u10_all_1_3 = data_u10_all_1[2:data_u10_all_1.shape[0] - 3, :, :, :]
    data_u10_all_1_4 = data_u10_all_1[3:data_u10_all_1.shape[0] - 2, :, :, :]
    data_u10_all_1_5 = data_u10_all_1[4:data_u10_all_1.shape[0] - 1, :, :, :]
    data_u10_all_1_6 = data_u10_all_1[5:data_u10_all_1.shape[0], :, :, :]
    data_u10_all_1_1 = torch.from_numpy(data_u10_all_1_1)
    data_u10_all_1_2 = torch.from_numpy(data_u10_all_1_2)
    data_u10_all_1_3 = torch.from_numpy(data_u10_all_1_3)
    data_u10_all_1_4 = torch.from_numpy(data_u10_all_1_4)
    data_u10_all_1_5 = torch.from_numpy(data_u10_all_1_5)
    data_u10_all_1_6 = torch.from_numpy(data_u10_all_1_6)
    data_u10_all_1_new = torch.cat(
        [data_u10_all_1_1, data_u10_all_1_2, data_u10_all_1_3, data_u10_all_1_4, data_u10_all_1_5,
         data_u10_all_1_6],
        dim=3)
    #################################################################################
    data_v10 = pd.read_csv(path_v10, header=None)
    # data_v10_reduced = 0.1*data_v10  # 数据处理为0.1倍
    data_v10_reduced = data_v10
    data_v10_all = create_dataset(data_v10_reduced, Height_wind)
    #data_v10_all_2 = interpolation(data_v10_all, Height_wind_chazhi, Wide_wind_chazhi)
    data_v10_all_3 = xuebian_dataset(data_v10_all)  # 削边线
    data_v10_all_1 = np.expand_dims(data_v10_all_3, 3)
    #################################################################################
    data_v10_all_1_1 = data_v10_all_1[0:data_v10_all_1.shape[0] - 5, :, :, :]
    data_v10_all_1_2 = data_v10_all_1[1:data_v10_all_1.shape[0] - 4, :, :, :]
    data_v10_all_1_3 = data_v10_all_1[2:data_v10_all_1.shape[0] - 3, :, :, :]
    data_v10_all_1_4 = data_v10_all_1[3:data_v10_all_1.shape[0] - 2, :, :, :]
    data_v10_all_1_5 = data_v10_all_1[4:data_v10_all_1.shape[0] - 1, :, :, :]
    data_v10_all_1_6 = data_v10_all_1[5:data_v10_all_1.shape[0], :, :, :]
    data_v10_all_1_1 = torch.from_numpy(data_v10_all_1_1)
    data_v10_all_1_2 = torch.from_numpy(data_v10_all_1_2)
    data_v10_all_1_3 = torch.from_numpy(data_v10_all_1_3)
    data_v10_all_1_4 = torch.from_numpy(data_v10_all_1_4)
    data_v10_all_1_5 = torch.from_numpy(data_v10_all_1_5)
    data_v10_all_1_6 = torch.from_numpy(data_v10_all_1_6)
    data_v10_all_1_new = torch.cat(
        [data_v10_all_1_1, data_v10_all_1_2, data_v10_all_1_3, data_v10_all_1_4, data_v10_all_1_5,
         data_v10_all_1_6],
        dim=3)
    data_wind = torch.cat([data_u10_all_1_new, data_v10_all_1_new], dim=3)  # 在axis=3拼接
    #################################################################################
    data_swh = pd.read_csv(path_swh, header=None)
    # data_swh_reduced = 0.1*data_swh  # 数据处理为0.1倍
    data_swh_reduced = data_swh
    data_swh_all = create_dataset(data_swh_reduced, Height_swh)
    data_swh_all_3 = xuebian_dataset(data_swh_all)  # 削边线
    data_swh_all_1 = np.expand_dims(data_swh_all_3, 3)
    #################################################################################
    data_swh_all_1_new = data_swh_all_1[5:data_swh_all_1.shape[0], :, :, :]
    data_swh_all_1_new = torch.from_numpy(data_swh_all_1_new)
    #################################################################################
    np.random.seed(num_disorder)
    data_wind_num = data_wind.shape[0]
    index = np.arange(data_wind_num)  # 生成下标
    np.random.shuffle(index)
    index = torch.from_numpy(index).long()
    index = index.unsqueeze(-1)
    index = index.unsqueeze(-1)
    index = index.unsqueeze(-1)
    index1 = index.expand(data_wind.shape)
    index = index.expand(data_swh_all_1_new.shape)
    #################################################################################
    data_wind_disorder = torch.gather(data_wind, dim=0, index=index1)  # (data_wind, dim=0, index=index)  # 按乱序下标取值
    # data_topo = pd.read_csv(path_topo, header=None)
    # data_topo = data2Ndata(data_topo, data_wind_disorder.shape[0])
    # data_topo = xuebian_dataset(data_topo)
    # data_topo = torch.from_numpy(np.expand_dims(data_topo, 3))
    # data_wind_disorder = torch.cat([data_wind_disorder, data_topo], dim=3)
    data_swh_disorder = torch.gather(data_swh_all_1_new, dim=0, index=index)
    data_train_number = int(len(data_wind) * (train_rate))
    data_test_number = len(data_wind) - data_train_number
    #################################################################################
    data_train_wind, data_test_wind = torch.split(data_wind_disorder, [data_train_number, data_test_number])
    data_train_swh, data_test_swh = torch.split(data_swh_disorder, [data_train_number, data_test_number])
    data_train_wind = data_train_wind.permute(0, 3, 1, 2)
    data_test_wind = data_test_wind.permute(0, 3, 1, 2)
    data_train_swh = data_train_swh.permute(0, 3, 1, 2)
    data_test_swh = data_test_swh.permute(0, 3, 1, 2)
    print("{} data for training.".format(data_train_number))
    print("{} data for validation.".format(data_test_number))

    return data_train_wind, data_test_wind, data_train_swh, data_test_swh


def train_one_epoch(model, optimizer, data, device, epoch, ):
    model.train()
    loss_function = torch.nn.L1Loss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_relative_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_train = tqdm(data, file=sys.stdout)

    for step, data_ws in enumerate(data_train):
        data_ws = data_ws.type(torch.FloatTensor)
        data_wind = data_ws[:, :12, :, :]
        data_swh = data_ws[:, -1, :, :]
        pred = model(data_wind.to(device))
        loss = loss_function(pred, data_swh.to(device))
        data_swh = data_swh.cuda()
        relative_loss = mean_relative_error(data_swh, pred)
        accu_relative_loss += relative_loss
        loss.backward()
        accu_loss += loss.detach()

        data_train.desc = "[train epoch {}] loss: {:.4f}, ACC: {:.4f}".format(epoch,
                                                                              accu_loss.item() / (step + 1),
                                                                              accu_relative_loss.item() / (step + 1))

        if not torch.isfinite(loss):
            print('\nWARNING: non-finite loss, ending training, check your dataset. \n', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_relative_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data, device, epoch):
    loss_function = torch.nn.L1Loss()

    model.eval()

    accu_loss = torch.zeros(1).to(device)
    accu_relative_loss = torch.zeros(1).to(device)
    data_train = tqdm(data, file=sys.stdout)
    for step, data_ws in enumerate(data_train):
        data_ws = data_ws.type(torch.FloatTensor)
        data_wind = data_ws[:, :12, :, :]
        data_swh = data_ws[:, -1, :, :]
        pred = model(data_wind.to(device))
        loss = loss_function(pred, data_swh.to(device))
        if (epoch == 499 or epoch == 999 or epoch == 1499) and step < 50:
            for i in range(8):
                pt = pred[i, :, :]
                dt = data_swh[i, :, :]
                pt = pd.DataFrame(pt.detach().cpu().numpy())
                dt = pd.DataFrame(dt.detach().cpu().numpy())
                ppath = str(epoch+1)+'predict_swh.csv'
                rpath = str(epoch+1)+'real_swh.csv'
                pt.to_csv(ppath, mode='a', header=False, index=False)
                dt.to_csv(rpath, mode='a', header=False, index=False)
        accu_loss += loss
        data_swh = data_swh.cuda()
        relative_loss = mean_relative_error(data_swh, pred)
        accu_relative_loss += relative_loss
        data_train.desc = "[valid epoch {}] loss: {:.4f}, ACC: {:.4f}".format(epoch,
                                                                              accu_loss.item() / (step + 1),
                                                                              accu_relative_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1), accu_relative_loss.item() / (step + 1)
