import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from vit_model import vit_base_patch16_224_in21k as create_model
from utils import read_split_data, train_one_epoch, evaluate
#  tensorboard.exe --logdir=


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    data_train_wind, data_test_wind, data_train_swh, data_test_swh = read_split_data(args.v10_path, args.u10_path,
                                                                                     args.topo_path, args.swh_path)

    data_train = torch.cat([data_train_wind, data_train_swh], dim=1)
    data_vali = torch.cat([data_test_wind, data_test_swh], dim=1)
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_data = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw)
    vali_data = torch.utils.data.DataLoader(data_vali,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=nw)

    model = create_model(has_logits=False).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5E-5, amsgrad=False)
    # 学习率衰减策略，lr=lr*x x=((1+cos(epoch*pi/epochs))/2)*(1-lrf)+lrf;
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data=train_data,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data=vali_data,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch+1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)  # 0.001
    parser.add_argument('--lrf', type=float, default=0.02)

    u10_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\26-36N,131.75-121.75W,2017(近岸)\ERA5\2017u10.csv'
    v10_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\26-36N,131.75-121.75W,2017(近岸)\ERA5\2017v10.csv'
    topo_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\32.75°N75.95°W-30.80°N74.00°W,2017\data_topo.csv'
    swh_path = r'E:\PythonProject\TRFexercise\VIT_Change\数据集\VIT数据集\26-36N,131.75-121.75W,2017(近岸)\ERA5\2017swh.csv'
    parser.add_argument('--u10-path', type=str,
                        default=u10_path)
    parser.add_argument('--v10-path', type=str,
                        default=v10_path)
    parser.add_argument('--topo-path', type=str,
                        default=topo_path)
    parser.add_argument('--swh-path', type=str,
                        default=swh_path)

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default=r'',
                        help='initial weights path')  # E:\PythonProject\TRFexercise\VIT_Change\预训练权重\model-1500.pth
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
