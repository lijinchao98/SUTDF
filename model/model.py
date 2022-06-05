import torch.nn.functional as F
import torch
from parts import * # 运行train时改为".parts"，model时候改为"parts"，路径问题
import sys
sys.path.append("..")
from data import dataset


class Encode_Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = Conv(1, 16)
        self.down2 = Conv(16, 32)
        self.fc = FC()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.fc(x2)

        return x3

class Detect_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.down1 = Conv(1, 16)
        self.down2 = Conv(16, 32)
        self.fc = FC()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.fc(x2)

        return x3

if __name__ == '__main__':

    # net = Encode_Net(n_channels=1, n_classes=2)
    # print(net)

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = Encode_Net()
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    HSI_dataset = dataset.HSI_Loader('../data/all_curve.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=32,
                                               shuffle=False)
    batch_size = 32
    for curve, label in train_loader:
        # 将数据拷贝到device中
        curve = curve.reshape(batch_size, 1, -1).to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)
        break
    print(curve.shape)
    # 使用网络参数，输出预测结果
    h = net(curve)
    print(h.shape)
