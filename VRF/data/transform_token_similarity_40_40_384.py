#!/usr/bin/python3
#coding=utf-8

import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, gt, token):
        for op in self.ops:
            image, mask, gt, token = op(image, mask, gt, token)
        return image, mask, gt, token

class RGBDCompose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, depth, mask):
        for op in self.ops:
            image, depth, mask = op(image, depth, mask)
        return image, depth, mask


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask, gt, token):
        image = (image - self.mean)/self.std
        # mask /= 255
        return image, mask, gt, token

class RGBDNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, depth, mask):
        image = (image - self.mean)/self.std
        depth = (depth - self.mean)/self.std
        mask /= 255
        return image, mask

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, gt, token):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        gt = cv2.resize(gt, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, gt, token

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, gt, token):
        H,W,_ = image.shape
        xmin  = np.random.randint(W-self.W+1)
        ymin  = np.random.randint(H-self.H+1)
        image = image[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask  = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        gt    = gt[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask, gt, token


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, gt, token):
        if np.random.randint(2)==1:
            image = image[:,::-1,:].copy()
            mask  =  mask[:,::-1,:].copy()
            gt = gt[:, ::-1, :].copy()
            token = token[:, ::-1, :].copy()

            # token = token.reshape(40, 40, 1600)
            # token = token[:, ::-1, :].copy()
            # token = token.reshape(1600, 40, 40)
            # token = token[:, :, ::-1].copy()
            # token = token.reshape(1600, 1600)
        return image, mask, gt, token



class ToTensor(object):
    def __call__(self, image, mask, gt, token):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        gt  = torch.from_numpy(gt)
        gt  = gt.permute(2, 0, 1)
        token  = torch.from_numpy(token)
        #token  = token.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), gt.mean(dim=0, keepdim=True), token

