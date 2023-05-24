from data_util.Object3D import *
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_DIR = os.path.join(ROOT_DIR,'data')

TRAIN_DIR = os.path.join(DATA_DIR,'dataflow/train')
TEST_DIR  = os.path.join(DATA_DIR,'dataflow/test')

def generate_dataset(obj3d,n_transform,filename,partition='train'):
        
        if partition=='train':
            obj3d.save(os.path.join(TRAIN_DIR, filename+'_0_0_.npy'))
            dir = TRAIN_DIR
        
        if partition=='test':
            obj3d.save(os.path.join(TEST_DIR, filename+'_0_0_.npy'))
            dir = TEST_DIR

        for i in np.arange(1,n_transform+1,1):

            o2 = obj3d.copy(transform=True)
            o2.save(os.path.join(dir,filename+'_'+str(i)+'_'+str(int(o2.label[0,0]))+'_'))

def maison(L,l,h,h2,transform=False):
    #rectangle
    p1 = np.array([0,0,0])
    p2 = np.array([L,0,0])
    p3 = np.array([L,l,0])
    p4 = np.array([0,l,0])
    p5 = np.array([0,0,h])
    p6 = np.array([L,0,h])
    p7 = np.array([L,l,h])
    p8 = np.array([0,l,h])
    p9 = np.array([0,l/2,h+h2])
    p10 = np.array([L,l/2,h+h2])
    density = 500

    plane1 = Planes(p1,p4,p2,density)
    plane2 = Planes(p4,p1,p8,density)
    plane3 = Planes(p1,p2,p5,density)
    plane4 = Planes(p2,p6,p3,density)
    plane5 = Planes(p4,p8,p3,density)
    plane6 = Planes(p6,p5,p7,density)
    trian1 = Triangle(p10,p7,p6,density=1000)
    trian2 = Triangle(p9,p5,p8,density=1000)
    plane7 = Planes(p9,p10,p5,density)
    plane8 = Planes(p9,p10,p8,density)

    maison = Object3D([plane1,plane2,plane3,plane4,plane5,plane6,trian1,trian2,plane8,plane7])
    vizualize([maison.xyz])
    maison = Object3D([plane1,plane2,plane3,plane4,plane5,plane6,trian1,trian2,plane8,plane7],transform=True)
    vizualize([maison.xyz])
    # maison.save('maison.npy')
    # vizualize([plane1.xyz,plane2.xyz,plane3.xyz,plane4.xyz,plane5.xyz,plane6.xyz,trian1.xyz,trian2.xyz,plane8.xyz,plane7.xyz])
    # vizualize([trian1.xyz,trian2.xyz])

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
    generate_dataset(obj,nTransform,planeName,partition)



if __name__ == '__main__':
    n_planes = 16
    for i in range(n_planes):
         L = np.random.uniform(4,10)
         l = np.random.uniform(2,6)
         name = 'Plane'+str(i)
         single_plane(L,l,nTransform=4,planeName=name,partition='test')