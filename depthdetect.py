from __future__ import absolute_import, division, print_function

import os
import argparse
import time
from collections import namedtuple
import sys

import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

from .pytorch.bts_dataloader import BtsDataLoader, ToTensor
from .pytorch.models.bts_eigen_v2_pytorch_densenet161\
        .bts_eigen_v2_pytorch_densenet161 import BtsModel

class Depthdetect():

    def __init__(self, **kwargs):

        Argument = namedtuple(
            'Argument',
            ['model_name', 'encoder', 'data_path', 'filenames_file',
             'input_height', 'input_width', 'max_depth', 'checkpoint_path',
             'dataset', 'do_kb_crop', 'save_lpg', 'bts_size', 'mode']
            )
        self.args = Argument(
            model_name='bts_eigen_v2_pytorch_densenet161',
            encoder='densenet161_bts',
            data_path='amap_traffic_final_train_data',
            filenames_file='../train_test_inputs/eigen_test_files_with_gt.txt',
            input_height=768,
            input_width=1216,
            max_depth=100,
            checkpoint_path=kwargs.get('checkpoint_path', 'pytorch/models/'
                            'bts_eigen_v2_pytorch_densenet161/model'),
            dataset='kitti',
            do_kb_crop=False,
            save_lpg=False,
            bts_size=512,
            mode='test'
            )
        self.model = BtsModel(params = self.args)
        self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        checkpoint = torch.load(self.args.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.cuda()

        # exec("param=" + sys.argv[1])
        # self.dict = {"model_name":}

    def get_num_lines(file_path):
        f = open(file_path, 'r')
        lines = f.readlines()
        f.close()
        return len(lines)

    def estimate(self, image):
        dataset = "kitti"

        """Test function."""
     

        # # num_test_samples = len(os.listdir(args.data_path))#########
        # ds_lines = os.listdir(args.data_path)
        # ds_lines.sort()
        # res_data = '/home/jiamingjie/python_program/depth_amap_traffic_final_train_data'

        # for i in ds_lines:
        #     args.data_path = args.data_path + '/' + i
        #     num_test_samples = len(os.listdir(args.data_path))
        #     dataloader = BtsDataLoader(args, 'test')
        #     lines = os.listdir(args.data_path)
        #     print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))
        # args.data_path = args.data_path+'/'+i
        # num_test_samples = len(os.listdir(args.data_path))
        focal = 721.5377
        image = cv2.resize(image, (1216,768)).astype(np.float32) / 255.0
        sample = {'image': image, 'focal': torch.tensor(focal)}
        sample = ToTensor('test')(sample)

        # lines = os.listdir(args.data_path)
        # print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))
        pred_depths = []
        pred_8x8s = []
        pred_4x4s = []
        pred_2x2s = []
        pred_1x1s = []
            #
            # start_time = time.time()
        with torch.no_grad():
            # print(sample['image'])
            image = Variable(sample['image'].unsqueeze(0).cuda())
            focal = Variable(sample['focal'].unsqueeze(0).cuda())
            # Predict
            lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est = \
                self.model(image, focal)
            pred_depths.append(depth_est.cpu().numpy().squeeze())
            pred_8x8s.append(lpg8x8[0].cpu().numpy().squeeze())
            pred_4x4s.append(lpg4x4[0].cpu().numpy().squeeze())
            pred_2x2s.append(lpg2x2[0].cpu().numpy().squeeze())
            pred_1x1s.append(reduc1x1[0].cpu().numpy().squeeze())

            # # elapsed_time = time.time() - start_time
            # print('Elapesed time: %s' % str(elapsed_time))
            # print('Done.')

            # save_name = res_data

            # print('Saving result pngs..')
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

            # if not os.path.exists(save_name + '/' + i):
            #     os.mkdir(save_name + '/' + i)
            # for s in tqdm(range(num_test_samples)):
            # if args.dataset == 'kitti':
            #     # date_drive = lines[s].split('/')[1]
            #     filename_pred_png = save_name + '/' + i + '/' + lines[s].replace(
            #         '.jpg', '.png')
            #     # filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
            #     #     -1].replace('.jpg', '.png')
            #     # filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
            # elif args.dataset == 'kitti_benchmark':
            #     filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg',
            #                                                                                          '.png')
            #     filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg',
            #                                                                                           '.png')
            #     filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
            # else:
            #
            #     scene_name = lines[s]
            #     print(scene_name)
            #     filename_pred_png = save_name + '/raw/' + scene_name.replace('.jpg',
            #                                                                  '.png')  # + '_' + lines[s].split()[0].split('/')[1].replace(
            #     # '.jpg', '.png')
            #     print(filename_pred_png)
            #     filename_cmap_png = save_name + '/cmap/' + scene_name  # + '_' + lines[s].split()[0].split('/')[1].replace(
            #     # '.jpg', '.png')
            #     filename_gt_png = save_name + '/gt/' + scene_name  # + '_' + lines[s].split()[0].split('/')[1].replace(
            #     #  '.jpg', '.png')
            #     filename_image_png = save_name + '/rgb/' + scene_name  # + '_' + lines[s].split()[0].split('/')[1]
            # # print(cv2.imread(filename_pred_png))
            # rgb_path = os.path.join(args.data_path, './' + lines[s])  # .split()[0])
            #
            # # image = cv2.imread(rgb_path)
            # if args.dataset == 'nyu':
            #     gt_path = os.path.join(args.data_path, './' + lines[s])  # .split()[1])
            #     print("asdfasdfasdgsadg")
            #     print(gt_path)
            #     gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            #     gt[gt == 0] = np.amax(gt)

            pred_depth = pred_depths[0]
            pred_8x8 = pred_8x8s[0]
            pred_4x4 = pred_4x4s[0]
            pred_2x2 = pred_2x2s[0]
            pred_1x1 = pred_1x1s[0]
            if dataset == 'kitti' or dataset == 'kitti_benchmark':
                pred_depth_scaled = pred_depth * 256.0
            else:
                pred_depth_scaled = pred_depth * 1000.0

            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            #     cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            #
            #     if args.save_lpg:
            #         cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            #         if args.dataset == 'nyu':
            #             plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
            #             pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
            #             plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
            #             pred_8x8_cropped = pred_8x8[10:-1 - 9, 10:-1 - 9]
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8_cropped), cmap='Greys')
            #             pred_4x4_cropped = pred_4x4[10:-1 - 9, 10:-1 - 9]
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4_cropped), cmap='Greys')
            #             pred_2x2_cropped = pred_2x2[10:-1 - 9, 10:-1 - 9]
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2_cropped), cmap='Greys')
            #             pred_1x1_cropped = pred_1x1[10:-1 - 9, 10:-1 - 9]
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1_cropped), cmap='Greys')
            #         else:
            #             plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_8x8.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_8x8), cmap='Greys')
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_4x4.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_4x4), cmap='Greys')
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_2x2.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_2x2), cmap='Greys')
            #             filename_lpg_cmap_png = filename_cmap_png.replace('.png', '_1x1.png')
            #             plt.imsave(filename_lpg_cmap_png, np.log10(pred_1x1), cmap='Greys')
            # args.data_path = '/home/jiamingjie/python_program/amap_t
        return pred_depth_scaled



