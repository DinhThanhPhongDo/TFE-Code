import open3d as o3d
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR,'src')
DATA_DIR = os.path.join(BASE_DIR,'data')

def get_lst_files(typ,dir):
    lst_dir = os.listdir(dir)
    ply_files = [file for file in lst_dir if file.split('.')[-1]==typ]
    return ply_files


def get_data(ply_files,noise_std=0.03,pcd_param = 'uniform', pcd_pts=20000,display = False):
    for ply_file in ply_files:
        print(ply_file)

        mesh = o3d.io.read_triangle_mesh(os.path.join(SRC_DIR,ply_file))
        mesh.compute_vertex_normals()
        if pcd_param == 'uniform':
            pcd = mesh.sample_points_uniformly(number_of_points=pcd_pts)
        elif pcd_param == 'poisson':
            pcd = mesh.sample_points_poisson_disk(number_of_points=pcd_pts, init_factor=5)
        else:
            assert True, 'wrong pcd_param'
        
        pts   = np.asarray(pcd.points)
        noise = np.random.normal(0, noise_std, size=(len(pts), 3))
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts+noise)

        filename = ply_file.split('.')[0]
        np.save(os.path.join(DATA_DIR,filename+'.npy'),pts+noise)
        o3d.io.write_point_cloud(os.path.join(DATA_DIR,filename+'.pcd'), pcd2)
        if display:
            mesh.paint_uniform_color([211/255, 211/255, 211/255])
            o3d.visualization.draw_geometries([mesh],
                                  zoom=1.6,
                                  front=[1, 1, 1.5],
                                  lookat=[0, 0, 1.532],
                                  up=[-1, -1, 1.5])
            pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
            pcd2.paint_uniform_color([211/255,211/255,211/255])
            # o3d.visualization.draw_geometries([pcd])
            o3d.visualization.draw_geometries([pcd2],
                                  zoom=1.6,
                                  front=[1, 1, 1.5],
                                  lookat=[0, 0, 1.532],
                                  up=[-1, -1, 1.5])
    return

if __name__ == '__main__':
    ply_files = get_lst_files('ply',dir=SRC_DIR)
    # ply_files = [ply_files[-2]]
    get_data(ply_files,noise_std=0.03,pcd_param = 'uniform', pcd_pts=20000,display = True)
    pcd_files = get_lst_files('pcd',DATA_DIR)

    # lst_pcd = []
    # for pcd_file in pcd_files:
    #     pcd = o3d.io.read_point_cloud(os.path.join(DATA_DIR,pcd_file))
    #     lst_pcd.append(pcd)
    # o3d.visualization.draw_geometries(lst_pcd)






