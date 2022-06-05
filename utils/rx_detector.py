import time
import numpy as np
import matplotlib.pyplot as plt
from utils.add_target import add_target_pixel


def RX(HSI_curve, threshold):
    HSI_curve = HSI_curve
    mu = np.average(HSI_curve, axis=0)
    C = np.cov(HSI_curve.T)
    C_reverse = np.linalg.inv(C)
    result = np.zeros(len(HSI_curve))
    for i in range(len(HSI_curve)):
        distance =np.reshape((HSI_curve[i] - mu), (1, 176))
        rx_value = distance @ C_reverse @ distance.T
        result[i] = abs(rx_value)
        result[i] = (abs(rx_value) > threshold)
    return result

if __name__ == '__main__':

    # load origin HSI
    HSI_curve = np.load('../data/all_curve.npy')
    # calculate mean of background
    sum = np.zeros(176)
    for i in range(1001, 4000):
        sum += HSI_curve[i]
    background_mean = sum / 3000
    # set variable depth
    H = [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 10, 20, 50, 60, 100] # len(H)=17
    # replace background pixels with underwater-target pixel
    h_index = 0
    for i in range(250, 1531, 80): # end = start + step*(len(h)-1) + 1
        HSI_curve[i] = add_target_pixel(background_mean, H[h_index])
        h_index += 1
    # rx detect
    t1 = time.time()
    result = RX(HSI_curve, 270)
    t2 = time.time()
    print('rx detection time:', round(t2 - t1, 3), 's')
    result_map = np.reshape(result, (100, 100))
    print(result_map)
    plt.imshow(result_map)
    plt.show()