import sys
import os
import numpy as np

path = os.path.abspath(os.path.dirname(__file__))
if not path in sys.path:
    sys.path.append(path)
    
from Planes import *
from Vizualize import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data')

TRAIN_DIR = os.path.join(DATA_DIR,'train')
TEST_DIR  = os.path.join(DATA_DIR,'test')

class Object3D:
    def __init__(self,lst_object, allowed_rot = None, is_noisy = False):
        
        self.lst_object  = lst_object
        self.allowed_rot = allowed_rot
        self.is_noisy    = is_noisy

        n_pts = 0
        for object3d in lst_object:
            n_pts += len(object3d.xyz)
        
        self.xyz  = np.zeros((n_pts,3))
        self.label= np.zeros((n_pts,1))
        self.flow = np.zeros((n_pts,3))

        count = 0
        for object3d in lst_object:
            self.xyz[count: count+len(object3d.xyz), : ] = object3d.get_xyz()
            # self.flow[count: count+len(object3d.xyz), : ] = 0...
            count += len(object3d.xyz)

    def transform(self):
        count = 0
        for i,object3d in enumerate(self.lst_object):

            transform_id = np.random.choice(np.arange(3))

            if transform_id ==0:
                xyz1 = object3d.get_xyz()
                if self.is_noisy:
                    xyz2 = object3d.add_noise()
                xyz2 = object3d.get_xyz()


            if transform_id ==1:
                t  = np.random.uniform(0.5,0.15,size=(3,)) #0.05 up to 0.15
                t *= np.random.choice([-1,1])
                xyz1 = object3d.get_xyz()
                object3d.translate(t)
                if self.is_noisy:
                    xyz2 = object3d.add_noise()
                xyz2 = object3d.get_xyz()

            if transform_id ==2:
                angle  = np.random.uniform(1,10) # 1 up to  10
                angle *= np.random.choice([-1,1])
                if self.allowed_rot == None:
                    axis  = np.random.choice([0,1,2])
                else:
                    axis  = np.random.choice(self.allowed_rot[i])
                xyz1 = object3d.get_xyz()
                object3d.rotate(axis,angle)
                if self.is_noisy:
                    xyz2 = object3d.add_noise()
                xyz2 = object3d.get_xyz()
                

            self.xyz[count: count+len(object3d.xyz), : ]  = xyz2
            self.label[count: count+len(object3d.xyz),0]  = np.ones(len(object3d.xyz))*transform_id
            self.flow[count: count+len(object3d.xyz), : ]+= xyz1-xyz2
            count += len(object3d.xyz)
    def translate(self, t) :
        count = 0
        for object in self.lst_object:
            xyz1 = object.get_xyz()
            object.translate(t)
            xyz2 = object.get_xyz()
            self.flow[count: count+len(object.xyz), : ] += xyz1-xyz2
            self.xyz[count: count+len(object.xyz), : ]   = xyz2
            count += len(xyz1)

    def rotate(self,axis,angle) :
        count = 0
        for object in self.lst_object:
            xyz1 = object.get_xyz()
            object.rotate(axis,angle,origin=True)
            xyz2 = object.get_xyz()
            self.flow[count: count+len(object.xyz), : ] += xyz1-xyz2
            self.xyz [count: count+len(object.xyz), : ]  = xyz2
            count += len(xyz1)
    
    def reset_flow(self):
        self.flow = np.zeros(shape = self.flow.shape)

    def save(self,filename):
        xyzfl = np.concatenate((self.xyz,self.flow,self.label),axis=1)
        np.save(filename,xyzfl)
    
    def copy(self,allowed_rot=None,is_noisy=False):
        tmp_lst_object = []
        for object in self.lst_object:
            tmp_lst_object.append(object.copy())

        return Object3D(tmp_lst_object,allowed_rot,is_noisy)



    def add_noise(self,mean=0,std=0.03):
        count = 0
        for object in self.lst_object:
            xyz1 = object.get_xyz()
            object.add_noise(mean,std)
            xyz2 = object.get_xyz()
            self.flow[count: count+len(object.xyz), : ] += xyz1-xyz2
            self.xyz[count: count+len(object.xyz), : ]   = object.xyz
            count += len(xyz1)

def test1():
    p1 = np.array([0,0,0])
    p2 = np.array([0,1,0])
    p3 = np.array([1,0,0])
    p4=  np.array([0,0,0])
    p5=  np.array([0,0,1])
    p6=  np.array([0,1,0])
    p7=  np.array([0,0,0])
    p8=  np.array([1,0,0])
    p9=  np.array([0,0,1])
    rot = []
    plane0 = Planes(p4,p5,p6,density=500) #black
    rot.append([1,2])
    plane1 = Planes(p1,p2,p3,density=500) #red
    rot.append([0,2])
    plane2 = Planes(p7,p8,p9,density=500) #green
    rot.append([1])


    lst = [plane0,plane1,plane2]
    cube = Object3D(lst,allowed_rot=rot, is_noisy=False)
    # vizualize([plane0.xyz,plane1.xyz,plane2.xyz])
    # vizualize([cube.xyz])
    cube.transform()
    vizualize([cube.xyz + cube.flow]) 
    cube.transform()
    tr = np.copy(cube.xyz) #red
    cube.rotate(1,20) #green
    # cube.transform()
    # vizualize([plane0.xyz,plane1.xyz,plane2.xyz])
    # vizualize([cube.xyz])
    vizualize([cube.xyz + cube.flow, tr, cube.xyz])
if __name__ == '__main__':
    test1()




        

        

