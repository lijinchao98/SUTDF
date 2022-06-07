from utils.add_target import add_target_pixel
import numpy as np

if __name__ == '__main__':
        # read HSI all curve from npy file
        HSI_curve = np.load('../data/all_curve.npy')
        # set variable depth
        H = []
        h = 0.01
        while h<=2:
            H.append(h)
            h += 0.02
            h = float('%.2f'%(h))
        while h<=4:
            H.append(h)
            h += 0.2
            h = float('%.2f'%(h))
        while h<=20:
            H.append(h)
            h += 1
            h = float('%.2f'%(h))
        print(H)
        print(len(H))
        # replace background pixels with underwater-target pixel
        guide_data = np.zeros((100*len(H), 176))
        for j in range(100):
            delta = j*len(H)
            for i in range(len(H)):
                guide_data[i + delta] = add_target_pixel(H[i], j) # 对水背景的第j个像素点，添加这么多个深度
            print(j)
        label = np.ones(100*len(H))
        print(guide_data[0].shape)
        np.save('guide_curve.npy', guide_data )
        np.save('guide_label.npy', label)