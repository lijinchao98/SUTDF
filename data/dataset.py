import torch
from torch.utils.data import Dataset
import numpy as np
import random


class HSI_Loader(Dataset):

    def __init__(self, curve, label):
        self.all_curve = np.load(curve)
        self.h = np.load(label)

    def __getitem__(self, index):
        # 根据index读取pixel的光谱曲线
        pixel_curve = torch.tensor(self.all_curve[index, :])
        h = torch.tensor(self.h[index])

        return pixel_curve, h

    def __len__(self):
        # 返回训练集大小
        return len(self.all_curve)


if __name__ == "__main__":
    HSI_dataset = HSI_Loader('guide_curve.npy', 'guide_h.npy')
    print("数据个数：", len(HSI_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=32,
                                               shuffle=False)
    batch_size = 32
    for pixel_curve, label in train_loader:
        print(pixel_curve.reshape(batch_size, 1, -1).shape)
        print(label)
        break
