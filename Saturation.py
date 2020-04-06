#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:14:34 2020

@author: po-wei
"""


import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

"""
  PS算法
  主要是利用HSL顏色空間進行飽和度S和亮度L的上下限控制，對RGB空間進行補丁式調整。
  參考CSDN博客：https://blog.csdn.net/maozefa/article/details/1781208
"""

def PSAlgorithm(rgb_img, increment):
    img = rgb_img * 1.0
    img_min = img.min(axis=2)
    img_max = img.max(axis=2)
    img_out = img
    
    #獲取HSL空間的飽和度和亮度
    delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value/2.0
    
    # s = L<0.5 ? s1 : s2
    mask_1 = L < 0.5
    s1 = delta/(value)
    s2 = delta/(2 - value)
    s = s1 * mask_1 + s2 * (1 - mask_1)
    
    # 增量大於0，飽和度指數增強
    if increment >= 0 :
        # alpha = increment+s > 1 ? alpha_1 : alpha_2
        temp = increment + s
        mask_2 = temp >  1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        
        alpha = 1/alpha -1 
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
        
    # 增量小於0，飽和度線性衰減
    else:
        alpha = increment
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha
    
    img_out = img_out/255.0
    
    # RGB顏色上下限處理(小於0取0，大於1取1)
    mask_3 = img_out  < 0 
    mask_4 = img_out  > 1
    img_out = img_out * (1-mask_3)
    img_out = img_out * (1-mask_4) + mask_4
    
    return img_out


def Saturation_Enhance(img, increment):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_new = PSAlgorithm(img, increment)
    
    '''
    plt.figure("img_original")
    plt.imshow(img/255.0)
    plt.axis('off')

    plt.figure("img_saturate")
    plt.imshow(img_new)
    plt.axis('off')

    plt.show()      
    '''
    return img_new * 255

