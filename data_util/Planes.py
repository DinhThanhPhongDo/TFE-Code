import numpy as np
# from Vizualize import vizualize

class Planes:
    def __init__(self,p1,p2,p3,density,T=None):
        """
        Generate a rectangular point cloud.
        parameters
        ----------
        p1,p2,p3 : np.array of size (3,).
                   Represent 3 points of the rectangle such that the angle(p2-p1,p3-p1)=90 degree
        density  : float. 
                   Number of point per m^2
        T        : np.array of size (4,4)
                   Represent transformation matrix in homogeneous coordinate 
        """
        
        self.density = density
        area    = np.linalg.norm(p1-p2) * np.linalg.norm(p1-p3)
        n_pts   = int(area*density)
        lambda_x,lambda_y = np.random.uniform(low=0,high=1,size=(2,n_pts)) # \in [0,1]

        if isinstance(T, (np.ndarray, np.generic) ): # T != None

            self.p1,self.p2,self.p3 = p1,p2,p3
            self.T = T
            #generate pcd centered in (0,0,0)
            self.xyz = (self.p1 + np.outer(lambda_x,self.p2-self.p1))+ (np.outer(lambda_y,self.p3-self.p1))
            #apply transform T to the centered pcd
            self.xyz = np.dot(self.T[0:3,0:3],self.xyz.T).T + self.T[0:3,3]

        else:
            # a rectangle (p1,p2,p3) center of gravity is not always (0,0,0). Thus, we consider that 
            # the rectangle (p1,p2,p3) has an 'implicit translation' when we give (p1,p2,p3) in argument.
            # self.p1 is when we center p1. 
            # ex: p1 = T@ self.p1 

            #find the centroid (which is also the implicit translation vector)
            c1 = p1 + np.outer(lambda_x,p2-p1)+ np.outer(lambda_y,p3-p1)
            centroid = np.mean(c1,axis=0)
            
            # we can compute self.p1 whcih is a centered p1 and the implicit transformation matrix
            self.p1,self.p2,self.p3 = p1-centroid,p2-centroid,p3-centroid
            self.T = np.array([[1,0,0,centroid[0]],
                               [0,1,0,centroid[1]],
                               [0,0,1,centroid[2]],
                               [0,0,0,1   ]])
            # i generate a pcd centered and then apply the transformation to translate and finally obtain
            # the rectange (p1,p2,p3).
            # this part can be simplified
            self.xyz = p1 + np.outer(lambda_x,p2-p1)+ np.outer(lambda_y,p3-p1)-centroid     
            self.xyz = np.dot(self.T[0:3,0:3],self.xyz.T).T + self.T[0:3,3]

    def translate(self,t):
        """
        translate the rectangle with a vector t (np.array of size (3,))
        """
        self.xyz += t
        T = np.array([[1,0,0,t[0]],
                      [0,1,0,t[1]],
                      [0,0,1,t[2]],
                      [0,0,0,1   ]])
        self.T = np.dot(T,self.T)
    
    def rotate(self,axis,angle,origin=False):
        """
        Rotate a rectange around its center of gravity in OX (axis=0), OY (axis=1), OZ (axis=2). angle in degree
        if origin=True, then rotate around the origin (0,0,0). Otherwise, rotate around the centroid
        """
        #translate to the origin
        t = self.T[[0,1,2],3]
        centroid = np.mean(self.xyz,axis=0)
        if not origin:
            self.translate(-centroid)

        #rotation
        s = np.sin(angle*(np.pi/180),dtype=np.float64)
        c = np.cos(angle*(np.pi/180),dtype=np.float64)
        if axis==0:
            Rx = np.array([[ 1, 0, 0, 0],
                           [ 0, c,-s, 0],
                           [ 0, s, c, 0],
                           [ 0, 0, 0, 1]])
            self.xyz = np.dot(Rx[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Rx,self.T)
        if axis==1:
            Ry = np.array([[ c, 0, s, 0],
                           [ 0, 1, 0, 0],
                           [-s, 0, c, 0],
                           [ 0, 0, 0, 1]])
            self.xyz = np.dot(Ry[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Ry, self.T)
        if axis==2:
            Rz = np.array([[ c,-s, 0, 0],
                           [ s, c, 0, 0],
                           [ 0, 0, 1, 0],
                           [ 0, 0, 0, 1]])
            self.xyz = np.dot(Rz[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Rz, self.T)




        #translate as at the begining
        if not origin:
            self.translate(+centroid)

    def copy(self):
        tmp_p1,tmp_p2,tmp_p3,tmp_density,tmp_T = self.p1,self.p2,self.p3,self.density,self.T
        return Planes(tmp_p1,tmp_p2,tmp_p3,tmp_density,tmp_T)
    
    def get_xyz(self):
        return np.copy(self.xyz)
    
    def add_noise(self,mean=0,std=0.03):
        noise = np.random.normal(mean,std,size=self.xyz.shape)
        self.xyz = self.xyz + noise

class Triangle:
    def __init__(self,p1,p2,p3,density,T=None):
        """
        Generate a rectangular point cloud.
        parameters
        ----------
        p1,p2,p3 : 3 np.array of size (3,).
                   Represent 3 points of the triangle.
        density  : float. 
                   Number of point per m^2
        T        : np.array of size (4,4)
                   Represent transformation matrix in homogeneous coordinate 
        """
        
        self.density = density
        # https://math.stackexchange.com/questions/128991/how-to-calculate-the-area-of-a-3d-triangle
        area    = np.linalg.norm(np.cross((p2-p1),(p3-p1)))/2
        n_pts   = int(area*density)
        lambdas = np.random.uniform(low=0,high=1,size=(3,n_pts))# \in [0,1]
        lambdas /= np.sum(lambdas,axis=0)

        if isinstance(T, (np.ndarray, np.generic) ): # T != None

            self.p1,self.p2,self.p3 = p1,p2,p3
            self.T = T
            #generate pcd centered in (0,0,0)
            self.xyz = np.outer(lambdas[0],self.p1) + np.outer(lambdas[1],self.p2)+ (np.outer(lambdas[2],self.p3))
            #apply transform T to the centered pcd
            self.xyz = np.dot(self.T[0:3,0:3],self.xyz.T).T + self.T[0:3,3]

        else:
            # a rectangle (p1,p2,p3) center of gravity is not always (0,0,0). Thus, we consider that 
            # the rectangle (p1,p2,p3) has an 'implicit translation' when we give (p1,p2,p3) in argument.
            # self.p1 is when we center p1. 
            # ex: p1 = T@ self.p1 

            #find the centroid (which is also the implicit translation vector)
            c1 = np.outer(lambdas[0],p1) + np.outer(lambdas[1],p2)+ (np.outer(lambdas[2],p3))
            centroid = np.mean(c1,axis=0)
            
            # we can compute self.p1 whcih is a centered p1 and the implicit transformation matrix
            self.p1,self.p2,self.p3 = p1-centroid,p2-centroid,p3-centroid
            self.T = np.array([[1,0,0,centroid[0]],
                               [0,1,0,centroid[1]],
                               [0,0,1,centroid[2]],
                               [0,0,0,1   ]],dtype=np.float64)
            # i generate a pcd centered and then apply the transformation to translate and finally obtain
            # the rectange (p1,p2,p3).
            # this part can be simplified
            self.xyz = np.outer(lambdas[0],p1) + np.outer(lambdas[1],p2)+ (np.outer(lambdas[2],p3))-centroid     
            self.xyz = np.dot(self.T[0:3,0:3],self.xyz.T).T + self.T[0:3,3]

    def translate(self,t):
        """
        translate the rectangle with a vector t (np.array of size (3,))
        """
        self.xyz += t
        T = np.array([[1,0,0,t[0]],
                      [0,1,0,t[1]],
                      [0,0,1,t[2]],
                      [0,0,0,1   ]],dtype=np.float64)
        self.T = np.dot(T,self.T)
    
    def rotate(self,axis,angle,origin=False):
        """
        Rotate a rectange around its center of gravity in OX (axis=0), OY (axis=1), OZ (axis=2). angle in degree
        """
        #translate to the origin
        t = self.T[[0,1,2],3]
        centroid = np.mean(self.xyz,axis=0)
        if not origin:
            self.translate(-centroid)

        #rotation
        s = np.sin(angle*(np.pi/180))
        c = np.cos(angle*(np.pi/180))
        if axis==0:
            Rx = np.array([[ 1, 0, 0, 0],
                           [ 0, c,-s, 0],
                           [ 0, s, c, 0],
                           [ 0, 0, 0, 1]],dtype=np.float64)
            self.xyz = np.dot(Rx[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Rx,self.T)
        if axis==1:
            Ry = np.array([[ c, 0, s, 0],
                           [ 0, 1, 0, 0],
                           [-s, 0, c, 0],
                           [ 0, 0, 0, 1]],dtype=np.float64)
            self.xyz = np.dot(Ry[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Ry, self.T)
        if axis==2:
            Rz = np.array([[ c,-s, 0, 0],
                           [ s, c, 0, 0],
                           [ 0, 0, 1, 0],
                           [ 0, 0, 0, 1]],dtype=np.float64)
            self.xyz = np.dot(Rz[0:3,0:3],self.xyz.T).T
            self.T   = np.dot(Rz, self.T)




        #translate as at the begining
        if not origin:
            self.translate(+centroid)
        return

    
    def copy(self):
        tmp_p1,tmp_p2,tmp_p3,tmp_density,tmp_T = self.p1,self.p2,self.p3,self.density,self.T
        return Triangle(tmp_p1,tmp_p2,tmp_p3,tmp_density,tmp_T)
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
    plane0 = Planes(p4,p5,p6,density=500) #black
    plane1 = Planes(p1,p2,p3,density=500) #red
    plane2 = Planes(p7,p8,p9,density=500) #green
    plane3 = plane2.copy()
    plane3.rotate(axis=0,angle=30)
    plane4 = plane3.copy()
    plane4.translate(np.array([0,1,0]))
    vizualize([plane0.xyz,plane1.xyz,plane2.xyz,plane3.xyz,plane4.xyz])

    
def test2(L,l,h,h2,transform=False):
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

    if transform==True:
        plane1.rotate(axis=1,angle=10)
        plane2.hole([[1/2,1/2],1/3])
        plane8.hole([[1/3,2/3],[1/3,2/3]])
        plane2.translate(np.array([-.3,0.2,0.1]))
        trian1.rotate(axis=2,angle=10)
        trian2.translate(np.array([-0.2,-0.2,+0.3]))
        plane8.translate(np.array([0,0,0.5]))
        plane8.rotate(axis=0,angle=5)


    vizualize([plane1.xyz,plane2.xyz,plane3.xyz,plane4.xyz,plane5.xyz,plane6.xyz,trian1.xyz,trian2.xyz,plane8.xyz,plane7.xyz])
    # vizualize([trian1.xyz,trian2.xyz])
    return

def test2(L,l,h,h2,transform=False):
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

    if transform==True:
        plane1.rotate(axis=1,angle=10)
        # plane2.hole([[1/2,1/2],1/3])
        #plane2.hole([[3/4,1],[1/2,1]])
        plane2.translate(np.array([-.3,0.2,0.1]))
        trian1.rotate(axis=2,angle=10)
        trian2.translate(np.array([-0.2,-0.2,+0.3]))
        plane8.translate(np.array([0,0,0.5]))
        plane8.rotate(axis=0,angle=30)


    vizualize([plane1.xyz,plane2.xyz,plane3.xyz,plane4.xyz,plane5.xyz,plane6.xyz,trian1.xyz,trian2.xyz,plane8.xyz,plane7.xyz])
    # vizualize([trian1.xyz,trian2.xyz])
    return
if __name__ =='__main__':
    # test1()
    # test2(L=3,l=2,h=1,h2=2,transform=False)
    test2(L=3,l=2,h=1,h2=2,transform=False)
    x = np.array([0,0,0])
    p1 = np.array([1,0,0])

