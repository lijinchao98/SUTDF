import numpy as np

HSI_curve = np.load('../data/all_curve.npy')

HSI_curve  = HSI_curve /10  # 调整数值大小
np.save('../data/all_curve.npy', HSI_curve) # 保存

print(HSI_curve.shape)
print(type(HSI_curve[0]))
