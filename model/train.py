from model import Encode_Net
from data.dataset import HSI_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
from utils.log import Logger
import torch.nn.functional as F
from utils.reconstruct import add_target_pixel
import numpy as np

sys.stdout = Logger()

def train_net(net, device, curve, label, epochs=1000, batch_size=1024, lr=0.00001):
    # 加载训练集
    Guide_dataset = HSI_Loader(curve, label)
    train_loader = torch.utils.data.DataLoader(dataset=Guide_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义Adam算法
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # 交叉熵 loss
    criterion = torch.nn.CrossEntropyLoss()  # 相当于LogSoftmax+NLLLoss
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for curve, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            curve = curve.reshape(len(curve), 1, -1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            # 使用网络参数，输出预测结果
            out = net(curve)
            # decoder
            # r = torch.zeros(len(h), 176).to(device=device) # 32 x 176
            # for i in range(len(h)):
            #     r[i] = add_target_pixel(h[i])
            # """
            # r, label都是[32, 176]
            # """
            # 定义SA光谱角损失函数
            # def SALoss(r, label):
            #     # 下面得到每行的二范数，也是再用哈达玛积相乘
            #     r_l2_norm = torch.norm(r, p=2, dim=1) # [1024]
            #     label_l2_norm = torch.norm(label, p=2, dim=1) # [1024]
            #     # r*label,对应元素乘,hadamard积,[1024,176],然后，每行求和torch.sum(r*label,dim=1)
            #     # 这样得到的是“向量r与向量label的内积”
            #     SALoss =  torch.sum(
            #         torch.acos(torch.sum(r*label, dim=1)/(r_l2_norm*label_l2_norm))
            #     ) # acos括号内为[1024]
            #     SALoss /= math.pi * len(r) # 除以pi归一化到[0,1]，除以batch_size平均一下
            #     return SALoss
            # 定义DVC深度值限制损失函数
            def DVCLoss(h):
                DVCLoss = abs(h).sum()/len(h)
                return DVCLoss
            # 计算Loss
            loss = criterion(out, label)
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_net.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        print(f'epoch:{epoch}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，曲线单通道1，分类为2。
    net = Encode_Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    curve = '../data/train_data.npy'
    label = '../data/train_label.npy'
    train_net(net, device, curve, label)