from data_util.Object3D import *
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
CLS_DATA_DIR = os.path.join(ROOT_DIR,'data\cls')
SEG_DATA_DIR = os.path.join(ROOT_DIR,'data\seg')


def generate_dataset(obj3d,n_transform,filename,data_dir,partition='train',allowed_rot=None,add_noise=False):
        
        if partition=='train':
            dir = os.path.join(data_dir,'train')
            obj3d.save(os.path.join(dir, filename+'_0_0_.npy'))
        
        if partition=='test':
            dir = os.path.join(data_dir,'test')
            obj3d.save(os.path.join(dir, filename+'_0_0_.npy'))

        for i in np.arange(1,n_transform+1,1):

            o2 = obj3d.copy(transform=True,allowed_rot=allowed_rot,add_noise=add_noise)
            o2.save(os.path.join(dir,filename+'_'+str(i)+'_'+str(int(o2.label[0,0]))+'_'))


def single_plane(L,l,nTransform = 10,planeName='Plane0',partition='test',add_noise=False):
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

    plane = Object3D([plane1],transform=False)

    
    # plane.translate(t)
    #obj.rotate(axis,angle) # do not work!

    generate_dataset(plane,nTransform,planeName,data_dir=CLS_DATA_DIR,partition=partition,allowed_rot=[[0,1]],add_noise=add_noise)

def room(L,l,h,h2,nTransform = 10,planeName='room0',partition='test',add_noise=False):
    #rectangle
    p1 = np.array([0,0,0])
    p2 = np.array([L,0,0])
    p3 = np.array([L,l,0])
    p4 = np.array([0,l,0])
    p5 = np.array([0,0,h])
    p6 = np.array([L,0,h])
    p7 = np.array([L,l,h])
    p8 = np.array([0,l,h])
    density = 500

    rot = []
    plane1 = Planes(p1,p4,p2,density)
    rot.append([0,1])
    plane2 = Planes(p4,p1,p8,density)
    rot.append([1,2])
    plane3 = Planes(p1,p2,p5,density)
    rot.append([0,2])
    plane4 = Planes(p2,p6,p3,density)
    rot.append([1,2])
    plane5 = Planes(p4,p8,p3,density)
    rot.append([0,2])
    plane6 = Planes(p6,p5,p7,density)
    rot.append([0,1])

    maison = Object3D([plane1,plane2,plane3,plane4,plane5,plane6])
    t = np.random.uniform(-10,10,size=(3,))
    angle = np.random.uniform(-180,+180)
    axis  = np.random.choice([0,1,2])

    maison.translate(t)
    #maison.rotate(axis,angle)

    generate_dataset(maison,nTransform,planeName,SEG_DATA_DIR, partition,allowed_rot=rot,add_noise=add_noise)

if __name__ == '__main__':
    
    n_planes = 6
    for i in range(n_planes):
         L = np.random.uniform(4,10)
         l = np.random.uniform(2,6)
         name = 'Plane'+str(i)
         single_plane(L,l,nTransform=4,planeName=name,partition='test',add_noise=True)
    
    # n_rooms = 8
    # for i in range(n_rooms):
    #     L = np.random.uniform(4,10)
    #     l = np.random.uniform(2,6)
    #     h1 = np.random.uniform(2,5)
    #     h2=None
    #     name = 'room'+str(i)
    #     room(L,l,h1,h2,nTransform=4,planeName=name,partition='train',add_noise=True)