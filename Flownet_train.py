#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloaders.FlowNetDataloader import Object3DDataset
from models.FlowNet3D import FlowNet3D

# from tensorboardX import SummaryWriter
from tqdm import tqdm

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    # print(np.shape(acc1))
    # print(np.shape(mask))
    # print(np.shape(mask_sum))
    # acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    # acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)#[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def test_one_epoch(net, test_loader):
    net.eval()

    total_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    num_examples = 0
    for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
        pc1, pc2, color1, color2, flow, mask1, l, f = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
        loss = torch.mean(mask1 * torch.sum((flow_pred - flow) * (flow_pred - flow), -1) / 2.0)
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(flow_pred.detach().cpu().numpy(), flow.detach().cpu().numpy(), mask1.detach().cpu().numpy())
        total_epe    += epe_3d * batch_size
        total_acc3d  += acc_3d * batch_size
        total_acc3d_2+=acc_3d_2*batch_size
        # print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))

        total_loss += loss.item() * batch_size
        

    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples


def train_one_epoch(net, train_loader, opt):
    net.train()
    num_examples = 0
    total_loss = 0
    for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        pc1, pc2, color1, color2, flow, mask1, l, f = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()
        flow = flow.cuda().transpose(2,1).contiguous()
        mask1 = mask1.cuda().float()

        batch_size = pc1.size(0)
        opt.zero_grad()
        num_examples += batch_size
        flow_pred = net(pc1, pc2, color1, color2)

        test_shape = torch.sum((flow_pred - flow) ** 2, 1)
        # print(np.shape(test_shape))
        # print(np.shape(mask1))
        loss = torch.mean(mask1 * torch.sum((flow_pred - flow) ** 2, 1) / 2.0)
        loss.backward()

        opt.step()
        total_loss += loss.item() * batch_size

        # if (i+1) % 100 == 0:
        #     print("batch: %d, mean loss: %f" % (i, total_loss / 100 / batch_size))
        #     total_loss = 0
    return total_loss * 1.0 / num_examples

def train(log_dir, net, lr, momentum, train_loader, test_loader, epochs, use_sgd=False):
    if use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=lr * 100, momentum=momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)
    scheduler = StepLR(opt, 10, gamma = 0.7)

    best_test_loss = np.inf
    for epoch in range(epochs):
        print('==epoch: %d, learning rate: %f=='%(epoch, opt.param_groups[0]['lr']))
        train_loss = train_one_epoch(net, train_loader, opt)
        print('mean train EPE loss: %f'%train_loss)

        test_loss, epe, acc, acc_2 = test_one_epoch(net, test_loader)
        print('mean test loss: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f'%(test_loss, epe, acc, acc_2))
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            print('best test loss till now: %f'%test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'pretrained_model\model.current_best.t7')
            else:
                torch.save(net.state_dict(), 'pretrained_model\model.current_best.t7')
        
        scheduler.step()


def main(log_dir):

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # CUDA settings

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    # logs
    # os.system('cp generate_flows.py checkpoints' + '/' + 'exp' + '/' + 'generate_flows.py.backup')
    # os.system('cp model.py checkpoints' + '/' + 'exp' + '/' + 'model.py.backup')
    # os.system('cp data_util/myFlowDataloader.py checkpoints' + '/' + 'exp' + '/' + 'myFlowDataloader.py.backup')

    train_loader = DataLoader(Object3DDataset(npoints=4096,partition='train'),num_workers=8)
    test_loader  = DataLoader(Object3DDataset(npoints=4096,partition='test'))


    net = FlowNet3D(None).cuda()
    net.apply(weights_init)

    if True:
        net.load_state_dict(torch.load('pretrained_model\model.best.t7'), strict=False)

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")


    lr = 0.001
    momentum = 0.9
    epochs = 10
    train(log_dir,net,lr,momentum,train_loader,test_loader,epochs,use_sgd=False)


    print('FINISH')
    # boardio.close()



if __name__ == '__main__':
    main(log_dir=None)