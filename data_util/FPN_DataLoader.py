import os
import numpy as np
import warnings

from torch.utils.data import Dataset

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data')

TRAIN_DIR = os.path.join(DATA_DIR,'train')
TEST_DIR  = os.path.join(DATA_DIR,'test')


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point



class FPN_DataLoader(Dataset):
    def __init__(self, npoints = 2048, partition='train'):
        self.partition = partition
        self.npoints = npoints

        if self.partition=='train':
            self.data_dir = TRAIN_DIR
        else:
            self.data_dir = TEST_DIR
        
        self.samples_npy = np.array(os.listdir(self.data_dir))
        self.defauts     = np.zeros(len(self.samples_npy),dtype=np.int32)
        self.models_npy  = np.copy(self.samples_npy)

        for i,npy_file in enumerate(self.samples_npy):
            plane_id,model_id,default_id,npy = npy_file.split('_')
            self.defauts[i] = np.int32(default_id)
            model_id = '0'
            self.transform_id='0'
            self.models_npy[i] = plane_id+'_'+model_id+'_'+self.transform_id+'_.npy'

    def __getitem__(self, index):

        pcd1  = np.load(os.path.join(self.data_dir, self.models_npy[index])) #model  pcd
        pcd2 = np.load(os.path.join(self.data_dir, self.samples_npy[index])) #sample pcd

        pcd1 = farthest_point_sample(pcd1, self.npoints).astype(np.float32)
        pcd2 = farthest_point_sample(pcd2, self.npoints).astype(np.float32)
        color1 = np.zeros((self.npoints,3),dtype=np.float32)
        color2 = np.zeros((self.npoints,3),dtype=np.float32)

        plane_id,model_id,default_idss,npy = self.samples_npy[index].split('_')
        labels = np.int32(default_idss)

        # pcd1 = pcd1[:, :3]
        # pcd2 = pcd2[:, :3]

        # pos1_center = np.mean(pcd1, 0)
        # pcd1 -= pos1_center
        # pcd2 -= pos1_center

        return pcd1, pcd2, color1, color2, self.defauts[index]
    def __len__(self):
        return len(self.samples_npy)

if __name__ == '__main__':
    import torch

    # data = FPN_DataLoader(partition='train')
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    # for pc1,pc2,c1,c2, label in DataLoader:
    #     print(pc1.shape)
    #     print(label.shape)

    tensor = torch.tensor([[[1,2],[3,4],[5,6]]])
    print(tensor)
    print(tensor.shape)
    print(tensor.reshape(3,-1))
    print(tensor.reshape(3,-1).shape)
