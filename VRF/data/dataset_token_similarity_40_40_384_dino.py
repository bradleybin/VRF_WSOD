#!/usr/bin/python3
#coding=utf-8

import os
import os.path as osp
import cv2
import torch
import numpy as np
try:
    from . import transform_token_similarity_40_40_384
except:
    import transform_token_similarity_40_40_384
from torch.utils.data import Dataset, DataLoader
from numpy import load

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs    = kwargs
        # print('\nParameters...')
        # for k, v in self.kwargs.items():
        #     print('%-10s: %s'%(k, v))

        if 'ECSSD' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.15, 112.48, 92.86]]])
            self.std       = np.array([[[ 56.36,  53.82, 54.23]]])
        elif 'DUTS' in self.kwargs['datapath']:
            self.mean      = np.array([[[124.55, 118.90, 102.94]]])
            self.std       = np.array([[[ 56.77,  55.97,  57.50]]])
        elif 'DUT-OMRON' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.61, 121.86, 114.92]]])
            self.std       = np.array([[[ 58.10,  57.16,  61.09]]])
        elif 'MSRA-10K' in self.kwargs['datapath']:
            self.mean      = np.array([[[115.57, 110.48, 100.00]]])
            self.std       = np.array([[[ 57.55,  54.89,  55.30]]])
        elif 'MSRA-B' in self.kwargs['datapath']:
            self.mean      = np.array([[[114.87, 110.47,  95.76]]])
            self.std       = np.array([[[ 58.12,  55.30,  55.82]]])
        elif 'SED2' in self.kwargs['datapath']:
            self.mean      = np.array([[[126.34, 133.87, 133.72]]])
            self.std       = np.array([[[ 45.88,  45.59,  48.13]]])
        elif 'PASCAL-S' in self.kwargs['datapath']:
            self.mean      = np.array([[[117.02, 112.75, 102.48]]])
            self.std       = np.array([[[ 59.81,  58.96,  60.44]]])
        elif 'HKU-IS' in self.kwargs['datapath']:
            self.mean      = np.array([[[123.58, 121.69, 104.22]]])
            self.std       = np.array([[[ 55.40,  53.55,  55.19]]])
        elif 'SOD' in self.kwargs['datapath']:
            self.mean      = np.array([[[109.91, 112.13,  93.90]]])
            self.std       = np.array([[[ 53.29,  50.45,  48.06]]])
        elif 'THUR15K' in self.kwargs['datapath']:
            self.mean      = np.array([[[122.60, 120.28, 104.46]]])
            self.std       = np.array([[[ 55.99,  55.39,  56.97]]])
        elif 'SOC' in self.kwargs['datapath']:
            self.mean      = np.array([[[120.48, 111.78, 101.27]]])
            self.std       = np.array([[[ 58.51,  56.73,  56.38]]])
        else:
            #raise ValueError
            self.mean = np.array([[[0.485*256, 0.456*256, 0.406*256]]])
            self.std = np.array([[[0.229*256, 0.224*256, 0.225*256]]])

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


class Data(Dataset):
    def __init__(self, cfg):
        with open(cfg.datapath+'/'+cfg.mode+'.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                #imagepath = cfg.datapath + '/image/' + line.strip() + '.jpg'
                imagepath = cfg.datapath + '/image/' + line.strip()
                #maskpath  = cfg.datapath + '/scribble/'  + line.strip() + '.png'
                maskpath = cfg.datapath + '/scribble/' + line.strip().replace('.jpg','.png')

                gtpath = cfg.datapath + '/mask/' + line.strip().replace('.jpg', '.png')
                if cfg.mode == 'test':
                    maskpath = cfg.datapath + '/mask/' + line.strip().replace('.jpg','.png')

                tokenth = '../TokenCut/DUTS_train_token_similarity_40_40_384/' + line.strip().replace('.jpg','.npy')

                self.samples.append([imagepath, maskpath, gtpath, tokenth])

        if cfg.mode == 'train':
            self.transform = transform_token_similarity_40_40_384.Compose(transform_token_similarity_40_40_384.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform_token_similarity_40_40_384.Resize(320, 320),
                                                    transform_token_similarity_40_40_384.RandomHorizontalFlip(),
                                                    transform_token_similarity_40_40_384.RandomCrop(320, 320),
                                                    transform_token_similarity_40_40_384.ToTensor())
        elif cfg.mode == 'test':
            self.transform = transform_token_similarity_40_40_384.Compose(transform_token_similarity_40_40_384.Normalize(mean=cfg.mean, std=cfg.std),
                                                    transform_token_similarity_40_40_384.Resize(320, 320),
                                                    transform_token_similarity_40_40_384.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        imagepath, maskpath, gtpath, tokenth = self.samples[idx]
        #print('imagepath!!!!!!!!!!!', imagepath)
        image               = cv2.imread(imagepath).astype(np.float32)[:,:,::-1]
        mask                = cv2.imread(maskpath).astype(np.float32)[:,:,::-1]
        gt                  = cv2.imread(gtpath).astype(np.float32)[:,:,::-1]


        token = load(tokenth)

        H, W, C             = mask.shape
        image, mask, gt, token         = self.transform(image, mask, gt, token)

        trans_token = token
        trans_token = trans_token.permute(2, 0, 1)


        token = token.reshape(1600, 384)

        token_final = token @ token.transpose(0, 1)



        mask[mask == 0.] = 255.
        mask[mask == 2.] = 0.
        return image, mask, (H, W), maskpath.split('/')[-1], gt, token_final, trans_token



    def __len__(self):
        return len(self.samples)



