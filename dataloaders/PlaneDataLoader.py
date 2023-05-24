import os
import numpy as np
import warnings

from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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


class PlaneDataLoader(Dataset):
    def __init__(self, root, process_data = False):
        self.root = root
        self.process_data = process_data
        filenames     = [os.path.join(root, f) for f in os.listdir(root)]
        self.datapath = [(f.split("_")[-2],f) for f in filenames]

        self.list_of_points = [None] * len(self.datapath)
        self.list_of_labels = [None] * len(self.datapath)
        # tqdm(range(len(self.datapath)), total=len(self.datapath))
        # for index in range(len(self.datapath)):
        #     point_set = np.load(self.datapath[index][1])[:,3:6].astype(np.float32)
        #     self.list_of_points[index] = point_set
        #     cls = self.datapath[index][0]
        #     cls = np.array([cls]).astype(np.int32)
        #     self.list_of_labels[index] = cls

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        point_set = np.load(self.datapath[index][1])[:,:6].astype(np.float32)
        label = np.array([self.datapath[index][0]]).astype(np.int32)
        point_set = farthest_point_sample(point_set, 2048)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)

if __name__ == '__main__':
    import torch

    data = PlaneDataLoader('Flownet\data\dataflow/train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)