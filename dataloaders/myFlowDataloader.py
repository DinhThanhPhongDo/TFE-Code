#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/cls')

TRAIN_DIR = os.path.join(DATA_DIR,'train')
TEST_DIR  = os.path.join(DATA_DIR,'test')

class Object3DDataset(Dataset):
    def __init__(self,npoints=2048, partition='train'):
        self.npoints = npoints
        self.partition = partition

        if self.partition=='train':
            self.data_dir = TRAIN_DIR
        else:
            self.data_dir = TEST_DIR
        
        self.samples_npy = np.array(os.listdir(self.data_dir))

    def __getitem__(self, index):
        model  = np.load(os.path.join(self.data_dir, self.samples_npy[index].split('_')[0]+'_0_0_.npy'))
        sample = np.load(os.path.join(self.data_dir, self.samples_npy[index]))

        pos1 = model[:,:3].astype('float32')
        pos2 = sample[:,:3].astype('float32')
        color1 = np.zeros((len(pos1),3),dtype=np.float32)
        color2 = np.zeros((len(pos2),3),dtype=np.float32)
        flow  = sample[:,3:6].astype('float32')
        mask  = np.ones((len(pos2),1),dtype=np.float32)
        labels = model[:,6].astype('float32')

        if self.partition == 'train':
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx2, :]
            mask = mask[sample_idx2, :]
            labels= labels[sample_idx2]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask = mask[:self.npoints, :]
            labels= labels[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, color1, color2, flow, mask, labels, self.samples_npy[index]

    def __len__(self):
        return len(self.samples_npy)



if __name__ == '__main__':
    train = Object3DDataset(partition='train')
    for i,data in enumerate(train):#train:
        print(i)
        pc1,pc2,c1,c2,filename = data
        print(pc1.shape)
        print(pc2.shape)
        print(c1.shape)
        print(c2.shape)

