import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
img_root = '/home/jiamingjie/python_program/depth_amap_traffic_final_train_data'
location = 'location.csv'

R_means = []
G_means = []
B_means = []
R_stds = []
G_stds = []
B_stds = []
# root,dirs,files = os.walk(img_root)
a = os.listdir(img_root)
a.sort()
img_name = os.listdir(os.path.join(img_root,a[1]))
for i in a:
    print(i)

    for k  in os.listdir(os.path.join(img_root, i)):
        impath = os.path.join(os.path.join(img_root, i),k)
        img = cv2.imread(impath)
        img = np.asarray(img)

        im = img.astype(np.float32)
        im_R = im[:, :, 0]
        im_G = im[:, :, 1]
        im_B = im[:, :, 2]
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)
a = [R_means,G_means,B_means]
b = [R_stds,G_stds,B_stds]
mean = [0,0,0]
std = [0,0,0]
mean[0] = np.mean(a[0])
mean[1] = np.mean(a[1])
mean[2] = np.mean(a[2])
std[0] = np.mean(b[0])
std[1] = np.mean(b[1])
std[2] = np.mean(b[2])
print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0],mean[1],mean[2]))
print('数据集的RGB方差为\n[{},{},{}]'.format(std[0],std[1],std[2]))