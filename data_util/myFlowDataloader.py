#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data')

TRAIN_DIR = os.path.join(DATA_DIR,'train')
TEST_DIR  = os.path.join(DATA_DIR,'test')
class Object3DDataset(Dataset):
    def __init__(self, partition='train'):
        self.partition = partition

        if self.partition=='train':
            self.data_dir = TRAIN_DIR
        else:
            self.data_dir = TEST_DIR
        
        self.samples_npy = np.array(os.listdir(self.data_dir))
        self.models_npy  = np.copy(self.samples_npy)

        for i,npy_file in enumerate(self.samples_npy):
            plane_id,model_id,default_id,npy = npy_file.split('_')
            model_id = '0'
            self.transform_id='0'
            self.models_npy[i] = plane_id+'_'+model_id+'_'+self.transform_id+'_.npy'


    def __getitem__(self, index):

        model  = np.load(os.path.join(self.data_dir, self.models_npy[index]))
        sample = np.load(os.path.join(self.data_dir, self.samples_npy[index]))

        pos1 = model[:,:3].astype('float32')
        pos2 = sample[:,:3].astype('float32')
        color1 = np.zeros((len(pos1),3),dtype=np.float32)
        color2 = np.zeros((len(pos2),3),dtype=np.float32)
        labels = np.ones((len(pos2),1),dtype=np.float32)* int(self.transform_id)



        pos1 = pos1[:, :3]
        pos2 = pos2[:, :3]
        # color1 = color1[:, :]
        # color2 = color2[:, :]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        return pos1, pos2, color1, color2, labels, self.samples_npy[index]

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

