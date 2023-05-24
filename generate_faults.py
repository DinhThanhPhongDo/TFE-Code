from data_util.Object3D import *
from data_util.Vizualize import vizualize
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
CLS_DATA_DIR = os.path.join(ROOT_DIR,'data\cls')
SEG_DATA_DIR = os.path.join(ROOT_DIR,'data\seg')

def generate_dataset(obj3d,n_transform,filename,data_dir,partition='train',allowed_rot=None):
        
        if partition=='train':
            dir = os.path.join(data_dir,'train')
            obj3d.save(os.path.join(dir, filename+'_0_0_.npy'))
        
        if partition=='test':
            dir = os.path.join(data_dir,'test')
            obj3d.save(os.path.join(dir, filename+'_0_0_.npy'))

        for i in np.arange(1,n_transform+1,1):

            o2 = obj3d.copy(transform=True,allowed_rot=allowed_rot)
            o2.save(os.path.join(dir,filename+'_'+str(i)+'_'+str(int(o2.label[0,0]))+'_'))


def single_plane(L,l,nTransform = 10,planeName='Plane0',partition='test'):
    p1 = np.array([0,0,0])
    p2 = np.array([L,0,0])
    p3 = np.array([0,l,0])
    density = 500

    plane1 = Planes(p1,p3,p2,density)
    t = np.random.uniform(-10,10,size=(3,))
    angle = np.random.uniform(-180,+180)
    axis  = np.random.choice([0,1,2])

    plane1.translate(t)
    plane1.rotate(axis,angle)

    obj = Object3D([plane1],transform=False)
    
    # do not work!
    # obj.translate(t)
    #obj.rotate(axis,angle)

    generate_dataset(obj,nTransform,planeName,data_dir=CLS_DATA_DIR,partition='train',allowed_rot=[[1]])



if __name__ == '__main__':
    n_planes = 6
    for i in range(n_planes):
         L = np.random.uniform(4,10)
         l = np.random.uniform(2,6)
         name = 'Plane'+str(i)
         single_plane(L,l,nTransform=1,planeName=name,partition='test')
    