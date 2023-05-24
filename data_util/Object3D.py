import numpy as np
from data_util.Planes import *
from data_util.Vizualize import *
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data')

TRAIN_DIR = os.path.join(DATA_DIR,'train')
TEST_DIR  = os.path.join(DATA_DIR,'test')

class Object3D:
    def __init__(self,lst_object, transform=False, allowed_rot = None):
        n_pts = 0
        for object3d in lst_object:
            n_pts += len(object3d.xyz)
        self.lst_object = lst_object
        self.xyz  = np.zeros((n_pts,3))
        self.label= np.zeros((n_pts,1))
        self.flow = np.zeros((n_pts,3))

        if not transform:
            count = 0
            for object3d in lst_object:
                self.xyz[count: count+len(object3d.xyz), : ] = object3d.xyz
                # self.flow[count: count+len(object3d.xyz), : ] = 0...
                count += len(object3d.xyz)
        else:
            count = 0
            for i,object3d in enumerate(lst_object):

                transform_id = np.random.choice(np.arange(3))

                if transform_id ==0:
                    xyz1 = object3d.get_xyz()
                    xyz2 = object3d.get_xyz()

                if transform_id ==1:
                    t  = np.random.uniform(0.05,0.15,size=(3,))
                    t *= np.random.choice([-1,1])
                    xyz1 = object3d.get_xyz()
                    object3d.translate(t)
                    xyz2 = object3d.get_xyz()

                if transform_id ==2:
                    angle = np.random.uniform(1,5)
                    angle *= np.random.choice([-1,1])
                    if allowed_rot == None:
                        axis  = np.random.choice([0,1,2])
                    else:
                        axis  = np.random.choice(allowed_rot[i])
                    xyz1 = object3d.get_xyz()
                    object3d.rotate(axis,angle)
                    xyz2 = object3d.get_xyz()
                    

                self.xyz[count: count+len(object3d.xyz), : ] = object3d.xyz
                self.label[count: count+len(object3d.xyz),0] = np.ones(len(object3d.xyz))*transform_id
                self.flow[count: count+len(object3d.xyz), : ] = xyz1-xyz2
                count += len(object3d.xyz)

    def save(self,filename):
        xyzfl = np.concatenate((self.xyz,self.flow,self.label),axis=1)
        np.save(filename,xyzfl)
    
    def copy(self,transform,allowed_rot=None):
        tmp_lst_object = []
        for object in self.lst_object:
            tmp_lst_object.append(object.copy())

        return Object3D(tmp_lst_object,transform,allowed_rot)

    def translate(self, t) :
        count = 0
        for object in self.lst_object:
            xyz1 = self.xyz[count: count+len(object.xyz), : ]
            object.translate(t)
            xyz2 = self.xyz[count: count+len(object.xyz), : ]
            self.flow[count: count+len(object.xyz), : ] += xyz1-xyz2
            self.xyz[count: count+len(object.xyz), : ] = object.xyz
            count += len(object.xyz)

    def rotate(self, axis,angle) :
        count = 0
        for object in self.lst_object:
            xyz1 = self.xyz[count: count+len(object.xyz), : ]
            object.rotate(axis,angle,origin=True)
            xyz2 = self.xyz[count: count+len(object.xyz), : ]
            self.flow[count: count+len(object.xyz), : ] += xyz1-xyz2
            self.xyz[count: count+len(object.xyz), : ] = object.xyz
            count += len(object.xyz)



        

        

