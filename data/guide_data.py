from utils.add_target import add_target_pixel
import numpy as np

if __name__ == '__main__':
        # read HSI all curve from npy file
        HSI_curve = np.load('../data/all_curve.npy')
        # range of wavelength, obtain from raw file
        # set variable depth
        H = []
        h = 0
        while h<=2:
            H.append(h)
            h += 0.01
            h = float('%.2f'%(h))
        while h<=20.1:
            H.append(h)
            h += 0.1
            h = float('%.2f'%(h))
        print(H)
        # replace background pixels with underwater-target pixel
        print(len(H))
        guide_data = np.zeros((2*len(H), 176))
        for i in range(len(H)):
                guide_data[i] = add_target_pixel(H[i])
                guide_data[i+len(H)] = HSI_curve[10*i]
        label = np.zeros(2*len(H))
        label[0:len(H)] = 1
        print(label)
        print(guide_data[0].shape)
        np.save('guide_curve.npy', guide_data )
        np.save('guide_label.npy', label)