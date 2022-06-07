import numpy as np
import matplotlib.pyplot as plt

target = np.load('../data/guide_curve.npy')
target_label = np.load('../data/guide_label.npy')
water = np.load('../data/all_curve.npy')
water_label = np.zeros(10000)

# HSI_curve  = HSI_curve /10  # 调整数值大小
# np.save('../data/all_curve.npy', HSI_curve) # 保存
train_data = np.vstack((target, water))
train_label = np.hstack((target_label, water_label))
# print(HSI_curve.shape)
# print(HSI_curve[0])
np.save('../data/train_data.npy', train_data)
np.save('../data/train_label.npy', train_label)
print(train_data[1], train_label[1] )
