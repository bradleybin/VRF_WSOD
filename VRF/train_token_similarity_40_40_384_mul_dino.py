#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from data import dataset_token_similarity_40_40_384_dino
#from data import dataset_test_dino
from net_agg_dino import SCWSSOD
import logging as logger
from lscloss import *
import numpy as np
from tools import *
#from Evaluation.evaluator import Eval_thread
#from Evaluation.dataloader import EvalDataset

from torch.autograd import Variable
#import matplotlib.pyplot as plt



TAG = "vrf_model"
SAVE_PATH = "vrf_model"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum


def get_polylr(base_lr, last_epoch, num_steps, power):
    return base_lr * (1.0 - min(last_epoch, num_steps-1) / num_steps) **power


BASE_LR = 1e-5
MAX_LR = 1e-2
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
batch = 16
l = 0.3


def train(Dataset, Network):
    ## dataset
    cfg = Dataset.Config(datapath='data/DUTS', savepath=SAVE_PATH, mode='train', batch=batch, lr=1e-3, momen=0.9, decay=5e-4, epoch=40)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    ## network
    net = Network(cfg)

    # num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print("可训练参数的数量：", num_params)

    # print('model has {} parameters in total'.format(sum(x.numel() for x in net.parameters())))
    criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean')
    loss_lsc = LocalSaliencyCoherence().cuda()
    net.train(True)
    net.train()
    net.cuda()
    criterion.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    #sw = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)

    # -------------------------- training ------------------------------------
    # start_event.record()
    #
    # torch.cuda.empty_cache()
    # max_memory_allocated = torch.cuda.max_memory_allocated()
    # max_memory_cached = torch.cuda.max_memory_cached()

    for epoch in range(cfg.epoch):
        batch_idx = -1
        cnt = 0
        #mae = 0
        for i, data_batch in enumerate(loader):
            cnt = cnt + 1
            image, mask, _, _, groundtruth, token, trans_token = data_batch
            image, mask, token, trans_token = Variable(image.cuda()),  Variable(mask.cuda()),  Variable(token.cuda()),  Variable(trans_token.cuda())
            image, mask, token, trans_token = image.float(), mask.float(), token.float(), trans_token.float()


            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr  # for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1

            ######  saliency structure consistency loss  ######
            image_scale = F.interpolate(image, scale_factor=0.3, mode='bilinear', align_corners=True)
            trans_token_scale = F.interpolate(trans_token, scale_factor=0.3, mode='bilinear', align_corners=True)
            out2, out3, out4, out5 = net(image,trans_token, 'Train')
            out2_s, out3_s, out4_s, out5_s = net(image_scale, trans_token_scale, 'Train')
            out2_scale = F.interpolate(out2[:, 1:2], scale_factor=0.3, mode='bilinear', align_corners=True)
            loss_ssc = SaliencyStructureConsistency(out2_s[:, 1:2], out2_scale, 0.85)


            ######   label for partial cross-entropy loss  ######
            gt = mask.squeeze(1).long()
            bg_label = gt.clone()
            fg_label = gt.clone()
            bg_label[gt != 0] = 255
            fg_label[gt == 0] = 255

            ######   local saliency coherence loss (scale to realize large batchsize)  ######
            image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=True)
            sample = {'rgb': image_}
            out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss2 = loss_ssc + criterion(out2, fg_label) + criterion(out2, bg_label) + l * loss2_lsc  ## dominant loss

            ######  auxiliary losses  ######
            out3_ = F.interpolate(out3[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss3_lsc = loss_lsc(out3_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss3 = criterion(out3, fg_label) + criterion(out3, bg_label) + l * loss3_lsc
            out4_ = F.interpolate(out4[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss4_lsc = loss_lsc(out4_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss4 = criterion(out4, fg_label) + criterion(out4, bg_label) + l * loss4_lsc
            out5_ = F.interpolate(out5[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=True)
            loss5_lsc = loss_lsc(out5_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']
            loss5 = criterion(out5, fg_label) + criterion(out5, bg_label) + l * loss5_lsc




            # print('torch.max(token)', torch.max(token))
            # print('torch.min(token)', torch.min(token))
            token = (token - torch.min(token)) / (torch.max(token) - torch.min(token))
            out2_t = F.interpolate(out2[:, 1:2], scale_factor=0.125, mode='bilinear', align_corners=True)
            B, C, H, W = out2_t.shape

            out2_t1 = out2_t.reshape(B, H * W, 1)
            f_token1 =  out2_t1 * token #A and V

            out2_t2 = out2_t.reshape(B, 1, H * W)
            fore_A = f_token1 * out2_t2

            b_token1 = (1- out2_t1) * token #B and V
            back_B = b_token1 * (1 - out2_t2)

            loss_global = 2 - torch.sum(fore_A)/torch.sum(f_token1) - torch.sum(back_B)/torch.sum(b_token1)
            ############



            out3_t = F.interpolate(out3[:, 1:2], scale_factor=0.125, mode='bilinear', align_corners=True)
            B, C, H, W = out3_t.shape

            out3_t1 = out3_t.reshape(B, H * W, 1)
            f_token1_3 =  out3_t1 * token #A and V

            out3_t2 = out3_t.reshape(B, 1, H * W)
            fore_A_3 = f_token1_3 * out3_t2

            b_token1_3 = (1- out3_t1) * token #B and V
            back_B_3 = b_token1_3 * (1 - out3_t2)

            loss_global_3 = 2 - torch.sum(fore_A_3)/torch.sum(f_token1_3) - torch.sum(back_B_3)/torch.sum(b_token1_3)
            #############



            out4_t = F.interpolate(out4[:, 1:2], scale_factor=0.125, mode='bilinear', align_corners=True)
            B, C, H, W = out4_t.shape

            out4_t1 = out4_t.reshape(B, H * W, 1)
            f_token1_4 =  out4_t1 * token #A and V

            out4_t2 = out4_t.reshape(B, 1, H * W)
            fore_A_4 = f_token1_4 * out4_t2

            b_token1_4 = (1- out4_t1) * token #B and V
            back_B_4 = b_token1_4 * (1 - out4_t2)

            loss_global_4 = 2 - torch.sum(fore_A_4)/torch.sum(f_token1_4) - torch.sum(back_B_4)/torch.sum(b_token1_4)
            #############



            out5_t = F.interpolate(out5[:, 1:2], scale_factor=0.125, mode='bilinear', align_corners=True)
            B, C, H, W = out5_t.shape

            out5_t1 = out5_t.reshape(B, H * W, 1)
            f_token1_5 =  out5_t1 * token #A and V

            out5_t2 = out5_t.reshape(B, 1, H * W)
            fore_A_5 = f_token1_5 * out5_t2

            b_token1_5 = (1- out5_t1) * token #B and V
            back_B_5 = b_token1_5 * (1 - out5_t2)

            loss_global_5 = 2 - torch.sum(fore_A_5)/torch.sum(f_token1_5) - torch.sum(back_B_5)/torch.sum(b_token1_5)

            loss_global_all = 1 * loss_global + 0.8 * loss_global_3 + 0.6 * loss_global_4 + 0.4 * loss_global_5
            #loss_global_all = 0.25 * loss_global + 0.25 * loss_global_3 + 0.25 * loss_global_4 + 0.25 * loss_global_5


            #loss_global = (torch.sum(fore_loss) - torch.sum(back_loss))/1600



            # groundtruth = groundtruth / 255
            # mae = mae + torch.mean((out2[:, 1:2] - groundtruth).abs())


            ######  objective function  ######
            loss = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4 + 0.15 * loss_global_all
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



            # torch.cuda.empty_cache()
            # max_memory_allocated = max(max_memory_allocated, torch.cuda.max_memory_allocated())
            # max_memory_cached = max(max_memory_cached, torch.cuda.max_memory_cached())



            #sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            if batch_idx % 100 == 0:
                msg = '%s| %s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss2_lsc=%.6f | loss_global=%.6f' % (SAVE_PATH, datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss2_lsc.item(), loss_global.item())
                print(msg)
                logger.info(msg)


        if epoch > 28:
            if (epoch+1) % 1 == 0 or (epoch+1) == cfg.epoch:
                torch.save(net.state_dict(), cfg.savepath+'/vrf-'+str(epoch+1)+'.pt')
        #print('mae-----------------------------------------------: ', mae/cnt )
        # end_event.record()
        # elapsed_time_ms = start_event.elapsed_time(end_event)
        # elapsed_time_sec = elapsed_time_ms / 1000
        #
        # print("训练时间（毫秒）：", elapsed_time_ms)
        # print("训练时间（秒）：", elapsed_time_sec)
        #
        # print("GPU 内存峰值已分配：", max_memory_allocated / 1024 / 1024, "MB")
        # print("GPU 内存峰值已缓存：", max_memory_cached / 1024 / 1024, "MB")


if __name__=='__main__':
    # start_event = torch.cuda.Event(enable_timing=True)
    # end_event = torch.cuda.Event(enable_timing=True)

    train(dataset_token_similarity_40_40_384_dino, SCWSSOD)


