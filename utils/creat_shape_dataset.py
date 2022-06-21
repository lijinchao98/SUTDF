from PIL import Image, ImageOps
import numpy as np
import os
import cv2 as cv

input_file =  r'/Users/lijinchao/Desktop/a.txt'
with open(input_file, 'r') as f:
    data = f.read().splitlines()

for i in range(len(data)):
    origin_image = Image.open(data[i]) # 读取图片
    origin_image.thumbnail((100,100),2) # 变为100x100的
    bw = origin_image.convert('1') # 转为二值图
    reverse_bw = ImageOps.invert(bw)
    root = '/Users/lijinchao/Desktop'  # 保存地址
    path = root + '/rectangular/' + str(i) +'.jpeg'  # 保存地址
    try:
        reverse_bw.save(path, quality=95)
        print('图片保存成功，保存在' + root + "\n")
    except:
        print('图片保存失败')


