from utils.add_target import add_target_pixel
import numpy as np

if __name__ == '__main__':
        # read HSI all curve from npy file
        HSI_curve = np.load('../data/all_curve.npy')
        # range of wavelength, obtain from raw file
        # set variable depth
        H = [100,80,50,40,40,40, 2,2,2.8,3.1,3.2,3.3,3.5,3.6,3.8,4,5,2,2,2,2,2,20, 7,10,80, 80,80,5,80]
        print(len(H))
        # replace background pixels with underwater-target pixel
        for i in range(len(H)):
                HSI_curve[i] = add_target_pixel(H[i],1)
        np.save('test_curve.npy', HSI_curve )