#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:27:00 2020

@author: po-wei
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import argparse
from Saturation import *



def get_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-l", "--layer_name", default = "block5_conv1", type = str, help = "Kernel form which layser! Default : block5_conv1")
    parser.add_argument("-e", "--epochs", default = 150, type = int, help = "number of iterations to update input!")
    parser.add_argument("-s", "--step_size", default = 1., type = float, help = "Like learning rate!")
    parser.add_argument("-fi", "--first_filter_index", default = 0, type = int, help = "first kernel wanna show form the layer!")
    parser.add_argument("-ei", "--end_filter_index", default = 1, type = int, help = "end kernel wanna show form the layer!")
    parser.add_argument("-i", "--increment", default = 0.7, type = float, help = "Saturation increment! From -1 ~ 1")
    
    args = parser.parse_args()
    
    return args


def normal(img):
    if np.ndim(img) == 4:
        img = np.squeeze(img, axis=0)
    img_norm = (img - np.min(img)) * 255 /np.ptp(img)
    return img_norm.astype('uint8')

def hisEqulColor(img):
    print("channel numbers %d:"%(np.ndim(img)))
    if np.ndim(img) >= 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        ycrcb = cv2.merge(channels)
        img_hist = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)

    else:
        img_hist = cv2.equalizeHist(img)
    return img_hist

def filter_index_adjustment(layer_name, first_filter_index, end_filter_index):
    final_kernel_index = model.get_layer(layer_name).output.shape[-1]
    if first_filter_index < 0:
        print("First index is smaller than 0! Set to 0!")
        first_filter_index = 0
    elif first_filter_index >= final_kernel_index:
        print("First index is bigger than %d! Set to %d!"%(final_kernel_index-1, final_kernel_index-1))
        first_filter_index = final_kernel_index - 1
    if end_filter_index < 0:
        print("End index is smaller than first index! Set to %d!"%(final_kernel_index + 1))
        end_filter_index = first_filter_index + 1
    elif end_filter_index > final_kernel_index:
        print("End index is bigger than %d! Set to %d!"%(final_kernel_index, final_kernel_index))
        end_filter_index = final_kernel_index
    return first_filter_index, end_filter_index
        


model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

args = get_parser()
layer_name = args.layer_name
epochs = args.epochs
step_size = args.step_size
first_filter_index, end_filter_index = filter_index_adjustment(layer_name, 
                                                               first_filter_index = args.first_filter_index, 
                                                               end_filter_index = args.end_filter_index)
increment = args.increment

org_img_loc = './kernel_pic_org/%s'%(layer_name)
enh_img_loc = './kernel_pic_enhance/%s'%(layer_name)
if not os.path.exists(org_img_loc):    #先確認資料夾是否存在
    os.mkdir(org_img_loc)
    os.mkdir(enh_img_loc)


# Create a connection between the input and the target layer
submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])
# Initiate random noise
input_img_data = np.random.random((1, 224, 224, 3))
input_img_data = (input_img_data - 0.5) * 20 + 128.

# Cast random noise from np.float64 to tf.float32 Variable
input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

for i in range(first_filter_index, end_filter_index):
    # Initiate random noise
    input_img_data = np.random.random((1, 224, 224, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128.
    
    # Cast random noise from np.float64 to tf.float32 Variable
    input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

    # Iterate gradient ascents
    for e in range(epochs):
        with tf.GradientTape() as tape:
            outputs = submodel(input_img_data)
            loss_value = tf.reduce_mean(outputs[:, :, :, i])
        grads = tape.gradient(loss_value, input_img_data)
        normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
        input_img_data.assign_add(normalized_grads * step_size)
        print("第幾 %d 個kernel的第 %d 次迭代！"%(i, e))
    img_norm = normal(input_img_data.numpy())
    img_enhance = Saturation_Enhance(img_norm, increment)
    cv2.imwrite(org_img_loc + '/%s_%d_org.jpg'%(layer_name, i), img_norm)
    cv2.imwrite(enh_img_loc + '/%s_%d_enhance.jpg'%(layer_name, i), img_enhance)

'''
img_norm = normal(input_img_data.numpy())
img_hist = hisEqulColor(img_norm)
img_enhance = Saturation_Enhance(img_norm, increment)

cv2.imshow("img_norm", img_norm)
cv2.imshow("img_hist", img_hist)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./kernel_pic/%s_%d.jpg'%(layer_name, first_filter_index), img_enhance)
'''








