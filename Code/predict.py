import torch
import pandas as pd
import numpy as np
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import xuebian_dataset, interpolation, create_dataset, data2Ndata, mean_relative_error


def read_split_data(path_v10: str, path_u10: str, path_swh: str):
    Height_wind = 41
    Height_swh = 41
    data_u10 = pd.read_csv(path_u10, header=None)
    data_u10_reduced = data_u10
    data_u10_all = create_dataset(data_u10_reduced, Height_wind)
    data_u10_all_3 = xuebian_dataset(data_u10_all)
    data_u10_all_1 = np.expand_dims(data_u10_all_3, 3)
    ################################################################################
    num = 6
    data_u10_all_1_new = torch.from_numpy(data_u10_all_1[0:data_u10_all_1.shape[0] - num])
    for i in range(1, num):
        data_u10_t = torch.from_numpy(data_u10_all_1[i:data_u10_all_1.shape[0] - num + i])
        data_u10_all_1_new = torch.cat([data_u10_all_1_new, data_u10_t], dim=3)
    #################################################################################
    data_v10 = pd.read_csv(path_v10, header=None)
    data_v10_all = create_dataset(data_v10, Height_wind)
    data_v10_all_3 = xuebian_dataset(data_v10_all)
    data_v10_all_1 = np.expand_dims(data_v10_all_3, 3)
    #################################################################################
    data_v10_all_1_new = torch.from_numpy(data_v10_all_1[0:data_v10_all_1.shape[0] - num])
    for i in range(1, num):
        data_v10_t = torch.from_numpy(data_v10_all_1[i:data_v10_all_1.shape[0] - num + i])
        data_v10_all_1_new = torch.cat([data_v10_all_1_new, data_v10_t], dim=3)
    data_wind = torch.cat([data_u10_all_1_new, data_v10_all_1_new], dim=3)
    data_swh = pd.read_csv(path_swh, header=None)
    data_swh_all = create_dataset(data_swh, Height_swh)
    data_swh_all_3 = xuebian_dataset(data_swh_all)
    data_swh_all_1 = np.expand_dims(data_swh_all_3, 3)
    #################################################################################
    data_swh_all_1_new = torch.from_numpy(data_swh_all_1[5:data_swh_all_1.shape[0], :, :, :])
    data_swh = data_swh_all_1_new.permute(0, 3, 1, 2)
    data_wind = data_wind.permute(0, 3, 1, 2)
    return data_wind.float(), data_swh.float()


def main(v_path, u_path, swh_path, weight_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    loss_function = torch.nn.L1Loss()
    model = create_model(has_logits=False).to(device)
    accu_loss = torch.zeros(1).to(device)
    accu_relative_loss = torch.zeros(1).to(device)
    # load model weights
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    data_wind, data_swh = read_split_data(v_path, u_path, swh_path)
    with torch.no_grad():
        for i in range(data_wind.shape[0]):
            data = data_wind[i].unsqueeze(0)
            data_swh1 = data_swh[i].squeeze(0)
            predict = model(data.to(device))
            loss = loss_function(predict, data_swh1.to(device))
            accu_loss += loss
            data_swh1 = data_swh1.cuda()
            relative_loss = mean_relative_error(data_swh1, predict)
            accu_relative_loss += relative_loss
            predict = predict.cpu()
            predict = pd.DataFrame(predict)
            data_swh1 = pd.DataFrame(data_swh1.cpu())
            predict.to_csv('predict_swh.csv', mode='a', header=False, index=False)
            data_swh1.to_csv('real_swh.csv', mode='a', header=False, index=False)
            if i % 100 == 0 or i == data_wind.shape[0]:
                print("loss: {:.4f}, ACC: {:.4f}".format(accu_loss.item() / (i+1), accu_relative_loss.item() / (i+1)))


if __name__ == '__main__':
    u10_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\14.5-24.5N,165.5-155.5W,2017(大洋)\ERA5\2017u10.csv'
    v10_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\14.5-24.5N,165.5-155.5W,2017(大洋)\ERA5\2017v10.csv'
    Swh_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\14.5-24.5N,165.5-155.5W,2017(大洋)\ERA5\2017swh.csv'
    model_weight_path = r'E:\PythonProject\TRFexercise\VIT_Change\data&results\NEW\30-40N,165.5-155.5W,2017(大洋)\model-500.pth'
    main(v10_path, u10_path, Swh_path, model_weight_path)
