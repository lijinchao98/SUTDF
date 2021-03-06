import matplotlib.pyplot as plt
from model import Encode_Net
from data.dataset import HSI_Loader
import numpy as np
from torch import optim
import torch.nn as nn
import torch
import sys
import math
from utils.log import Logger
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
import torch.nn.functional as F
from utils.reconstruct import add_target_pixel
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(precision=2, threshold=float('inf'), sci_mode=False)
sys.stdout = Logger()


def pred_net(net, device, curve, label, batch_size=25):
    # 加载训练集
    Guide_dataset = HSI_Loader(curve, label)
    train_loader = torch.utils.data.DataLoader(dataset=Guide_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    # 测试模式
    # 加载模型参数
    net.load_state_dict(torch.load('best_model_net.pth', map_location=device))# 加载模型参数
    net.eval()
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.long)
        print(label)
        # 使用网络参数，输出预测结果
        out = net(curve)
        # r = torch.zeros(len(h), 176).to(device=device)  # 32 x 176
        # for i in range(len(h)):
        #     r[i] = add_target_pixel(h[i])

        # pltcurve = 20 # 要看哪条曲线
        # hh = h.view(100,100).detach().cpu().numpy()
        # 设置一个阈值，判断目标还是背景
        # threshold = 1.9
        # for i in range(100):
        #     for j in range(100):
        #         if hh[i,j] > threshold:
        #             hh[i,j] = 100
        # plt.imshow(hh)
        # print(hh)
        #
        # print(f'r:{r[pltcurve,:]}')
        # print(f'label:{label[pltcurve,:]}')

        def SALoss(r, label):
            # 下面得到每行的二范数，也是再用哈达玛积相乘
            r_l2_norm = torch.norm(r, p=2, dim=1)  # [1024]
            label_l2_norm = torch.norm(label, p=2, dim=1)  # [1024]
            # r*label,对应元素乘,hadamard积,[1024,176],然后，每行求和torch.sum(r*label,dim=1)
            # 这样得到的是“向量r与向量label的内积”
            SALoss = torch.sum(
                torch.acos(torch.sum(r * label, dim=1) / (r_l2_norm * label_l2_norm))
            )  # acos括号内为[1024]
            SALoss /= math.pi * len(r)  # 除以pi归一化到[0,1]，除以batch_size平均一下
            return SALoss

        # 计算loss
        loss = criterion(out, label)
        pred = torch.max(out, dim=1)
        print(pred)
        print(loss)
        print(out)
        # 计算相对误差百分比REP(relative error percentage)
        # REP = torch.abs(r-label)/label # REP [1024,176]
        # sum_REP = torch.where(torch.isinf(torch.sum(REP, 1)), torch.full_like(torch.sum(REP, 1), 1), torch.sum(REP, 1)) # inf换为1
        # mean_REP = torch.mean(sum_REP)
        # print(f'loss:{loss.item()}')
        # print(f'REP:{torch.sum(REP[pltcurve,:]).item()}')
        # print(f'mean_REP:{mean_REP.item()}')

        # 画图
        # wavelength = [398.10,401.30,404.50,407.60,410.80,414.00,417.20,420.40,423.60,426.80,430.00,
        #                   433.30,436.50,439.70,442.90,446.10,449.40,452.60,455.80,459.10,462.30,465.50,
        #                   468.80,472.00,475.30,478.50,481.80,485.10,488.30,491.60,494.90,498.10,501.40,
        #                   504.70,508.00,511.30,514.60,517.80,521.10,524.40,527.70,531.00,534.40,537.70,
        #                   541.00,544.30,547.60,550.90,554.30,557.60,560.90,564.30,567.60,570.90,574.30,
        #                   577.60,581.00,584.30,587.70,591.10,594.40,597.80,601.20,604.50,607.90,611.30,
        #                   614.70,618.10,621.40,624.80,628.20,631.60,635.00,638.40,641.80,645.30,648.70,
        #                   652.10,655.50,658.90,662.40,665.80,669.20,672.70,676.10,679.50,683.00,686.40,
        #                   689.90,693.30,696.80,700.20,703.70,707.20,710.60,714.10,717.60,721.10,724.60,
        #                   728.00,731.50,735.00,738.50,742.00,745.50,749.00,752.50,756.00,759.50,763.10,
        #                   766.60,770.10,773.60,777.10,780.70,784.20,787.80,791.30,794.80,798.40,801.90,
        #                   805.50,809.00,812.60,816.20,819.70,823.30,826.90,830.40,834.00,837.60,841.20,
        #                   844.80,848.40,852.00,855.60,859.20,862.80,866.40,870.00,873.60,877.20,880.80,
        #                   884.40,888.10,891.70,895.30,899.00,902.60,906.20,909.90,913.50,917.20,920.80,
        #                   924.50,928.10,931.80,935.50,939.10,942.80,946.50,950.20,953.80,957.50,961.20,
        #                   964.90,968.60,972.30,976.00,979.70,983.40,987.10,990.80,994.50,998.20,1002.00]
        # # 重构曲线和真实曲线
        # plt.figure()
        # plt.plot(wavelength, torch.squeeze(r[pltcurve,:]).detach().cpu().numpy(),
        #          label='reconstruct', color='r', marker='o', markersize=3)
        # plt.plot(wavelength, torch.squeeze(label[pltcurve,:]).detach().cpu().numpy(),
        #          label='real', color='b', marker='o', markersize=3)
        # plt.xlabel('band')
        # plt.ylabel('reflect value')
        # plt.legend()
        #
        # plt.show()
        break



if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Encode_Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    curve = '../data/test_curve.npy'
    label = '../data/guide_label.npy'
    pred_net(net, device, curve, label)
