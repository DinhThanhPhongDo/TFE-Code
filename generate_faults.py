from data_util.Object3D import *
import os
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
CLS_DATA_DIR = os.path.join(ROOT_DIR,'data\cls_noisy')
SEG_DATA_DIR = os.path.join(ROOT_DIR,'data\seg_noisy')


def generate_dataset(obj3d,n_transform,filename,data_dir,allowed_rot=None,add_noise=False):
        
        obj3d.save(os.path.join(data_dir, filename+'_0_0_.npy'))

        for i in np.arange(1,n_transform+1,1):

            o2 = obj3d.copy(allowed_rot=allowed_rot,is_noisy=add_noise)
            o2.transform()
            o2.save(os.path.join(data_dir,filename+'_'+str(i)+'_'+str(int(o2.label[0,0]))+'_'))


def single_plane(data_dir,L,l,nTransform = 10,planeName='Plane0',add_noise=False):
    p1 = np.array([0,0,0])
    p2 = np.array([L,0,0])
    p3 = np.array([0,l,0])
    density = 500

    plane1 = Planes(p1,p3,p2,density)
    plane = Object3D([plane1])


    t = np.random.uniform(-10,10,size=(3,))
    angle = np.random.uniform(-180,+180)
    axis  = np.random.choice([0,1,2])

    
    plane.translate(t)
    # If rotation performed, the rotation of the plane is no longer along axis
    plane.rotate(axis,angle)
    plane.reset_flow()

    generate_dataset(plane,nTransform,planeName,data_dir=data_dir,allowed_rot=[[0,1]],add_noise=add_noise)

def room(data_dir,L,l,h,h2,nTransform = 10,planeName='room0',add_noise=False):
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
    # If rotation performed, the rotation of the plane is no longer along axis
    maison.rotate(axis,angle)
    maison.reset_flow()

    generate_dataset(maison,nTransform,planeName,data_dir=data_dir,allowed_rot=rot,add_noise=add_noise)

def generate_planes(data_dir,n_planes,nTransform,add_noise):

    # delete previous planes
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # generate new one
    for i in range(n_planes):
        L = np.random.uniform(4,10)
        l = np.random.uniform(2,6)
        name = 'Plane'+str(i)
        single_plane(data_dir,L,l,nTransform=nTransform,planeName=name, add_noise=add_noise)

def generate_rooms(data_dir,n_rooms,nTransform,add_noise):
    # delete previous rooms
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    # generate new one
    for i in range(n_rooms):
        L = np.random.uniform(4,10)
        l = np.random.uniform(2,6)
        h1 = np.random.uniform(2,5)
        h2=None
        name = 'room'+str(i)
        room(data_dir, L,l,h1,h2,nTransform=nTransform,planeName=name,add_noise=add_noise)

if __name__ == '__main__':

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    DATA_DIR = os.path.join(ROOT_DIR,'data/')

    plane_dirs= [os.path.join(DATA_DIR,'cls/train'),
                 os.path.join(DATA_DIR,'cls/test'),
                 os.path.join(DATA_DIR,'cls_noisy/train'),
                 os.path.join(DATA_DIR,'cls_noisy/test')]
    room_dirs = [os.path.join(DATA_DIR,'seg/train'),
                 os.path.join(DATA_DIR,'seg/test'),
                 os.path.join(DATA_DIR,'seg_noisy/train'),
                 os.path.join(DATA_DIR,'seg_noisy/test')]
    
    n_pl = [64,16,64,16]
    n_tr = [ 4, 4, 4, 4]
    # n_pl = [4,1,4,1]
    # n_tr = [ 4, 4, 4, 4]
    noise= [False,False,True,True]

    for i in range(len(plane_dirs)):
        print('dir=%s'%(plane_dirs[i]))
        generate_planes(plane_dirs[i],n_pl[i],n_tr[i],noise[i])

    n_ro = [64,16,64,16]
    n_tr = [ 4, 4, 4, 4]
    # n_ro = [4,1,4,1]
    # n_tr = [ 4, 4, 4, 4]
    noise= [False,False,True,True]
    for i in range(len(room_dirs)):
        print('dir=%s'%(room_dirs[i]))
        generate_rooms(room_dirs[i],n_ro[i],n_tr[i],noise[i])