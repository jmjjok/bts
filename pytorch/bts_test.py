# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import numpy as np
import cv2
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
from bts_dataloader import *

import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from bts_dataloader import *
a = os.getcwd()

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_eigen_v2_pytorch_densenet161')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data',default = '/home/jiamingjie/python_program/amap_traffic_final_train_data')#数据集的data路径
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file',default = "../train_test_inputs/eigen_test_files_with_gt.txt")
parser.add_argument('--input_height', type=int, help='input height', default=768)
parser.add_argument('--input_width', type=int, help='input width', default=1216)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=100)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='./lib/bts/pytorch/models/bts_eigen_v2_pytorch_densenet161/model')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='kitti')
parser.add_argument('--do_kb   _crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)

for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'

    
    model = BtsModel(params=args)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    # num_test_samples = len(os.listdir(args.data_path))#########
    ds_lines = os.listdir(args.data_path)
    ds_lines.sort()
    res_data = '/home/jiamingjie/python_program/depth_amap_traffic_final_train_data'

    for i in ds_lines:
        args.data_path = args.data_path+'/'+i
        num_test_samples = len(os.listdir(args.data_path))
        dataloader = BtsDataLoader(args, 'test')
        lines = os.listdir(args.data_path)
        print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

        pred_depths = []
        pred_8x8s = []
        pred_4x4s = []
        pred_2x2s = []
        pred_1x1s = []

        start_time = time.time()
        with torch.no_grad():
            for _, sample in enumerate(tqdm(dataloader.data)):
                # print(sample['image'])
                image = Variable(sample['image'].cuda())
                focal = Variable(sample['focal'].cuda())
                # Predict
                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = model(image, focal)
                print("**********")
                print(lpg8x8)
                print("**********")
                print(lpg4x4)
                print("**********")
                print("**********")
                pred_depths.append(depth_est.cpu().numpy().squeeze())
                pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
                pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
                pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
                pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

        elapsed_time = time.time() - start_time
        print('Elapesed time: %s' % str(elapsed_time))
        print('Done.')

        save_name = res_data

        print('Saving result pngs..')
        # if not os.path.exists(os.path.dirname(save_name)):
        #     try:
        #         os.mkdir(save_name)
        #         os.mkdir(save_name + '/raw')
        #         os.mkdir(save_name + '/cmap')
        #         os.mkdir(save_name + '/rgb')
        #         os.mkdir(save_name + '/gt')
        #     except OSError as e:
        #         if e.errno != errno.EEXIST:
        #             raise

        if not os.path.exists(save_name+'/'+i):
            os.mkdir(save_name+'/'+i)
        for s in tqdm(range(num_test_samples)):
            if args.dataset == 'kitti':
                # date_drive = lines[s].split('/')[1]
                filename_pred_png = save_name+'/'+i+'/' + lines[s].replace(
                    '.jpg', '.png')
                # filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
                #     -1].replace('.jpg', '.png')
                # filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
            elif args.dataset == 'kitti_benchmark':
                filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
            else:

                scene_name = lines[s]
                print(scene_name)
                filename_pred_png = save_name + '/raw/' + scene_name.replace('.jpg','.png')# + '_' + lines[s].split()[0].split('/')[1].replace(
                   # '.jpg', '.png')
                print(filename_pred_png)
                filename_cmap_png = save_name + '/cmap/' + scene_name #+ '_' + lines[s].split()[0].split('/')[1].replace(
                   # '.jpg', '.png')
                filename_gt_png = save_name + '/gt/' + scene_name# + '_' + lines[s].split()[0].split('/')[1].replace(
                  #  '.jpg', '.png')
                filename_image_png = save_name + '/rgb/' + scene_name# + '_' + lines[s].split()[0].split('/')[1]
            # print(cv2.imread(filename_pred_png))
            rgb_path = os.path.join(args.data_path, './' + lines[s])#.split()[0])

            # image = cv2.imread(rgb_path)
            if args.dataset == 'nyu':
                gt_path = os.path.join(args.data_path, './' + lines[s])#.split()[1])
                print("asdfasdfasdgsadg")
                print(gt_path)
                gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
                gt[gt == 0] = np.amax(gt)

            pred_depth = pred_depths[s]
            pred_8x8 = pred_8x8s[s]
            pred_4x4 = pred_4x4s[s]
            pred_2x2 = pred_2x2s[s]
            pred_1x1 = pred_1x1s[s]
            print(pred_depth)
            if args.dataset == 'kitti' or args.dataset == 'kitti_benchmark':
                pred_depth_scaled = pred_depth * 256.0
            else:
                pred_depth_scaled = pred_depth * 1000.0

            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            if args.save_lpg:
                cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
                if args.dataset == 'nyu':
                    plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                    pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                    plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
                    pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
                    pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
                    pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
                    pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
                else:
                    plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
                    filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
                    plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
        args.data_path = '/home/jiamingjie/python_program/amap_traffic_final_train_data'
    
    return


if __name__ == '__main__':
    test(args)
