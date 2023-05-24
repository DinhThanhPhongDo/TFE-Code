import open3d as o3d
import numpy as np

def vizualize(xyzs):
    pcds = []
    for i,xyz in enumerate(xyzs):
        r,g,b = np.random.uniform(0,1),np.random.uniform(0,1),np.random.uniform(0,1)
        if i == 0:
            r,g,b = 0,0,0
        elif i==1:
            r,g,b = 1,0,0
        elif i==2:
            r,g,b = 0,1,0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.paint_uniform_color([r,g,b])
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)


    

