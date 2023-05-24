import open3d as o3d
import numpy as np
import os
from plotly.offline import iplot
from plotly import graph_objs as go
import plotly.figure_factory as ff


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_DIR = os.path.join(ROOT_DIR,'data')

def vizualize_sans_flow(model,sample):
    """
    draw in gray the model pcd and in green the shifted model
    """
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(model)
    pcd_model.paint_uniform_color([0.7, 0.7, 0.7])

    pcd_shift = o3d.geometry.PointCloud()
    pcd_shift.points = o3d.utility.Vector3dVector(sample)
    pcd_shift.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd_model,pcd_shift])

def vizualize(model,sample,flow):
    """
    draw in gray the model, in green the shifted model, in red the result obtained after the flow.
    """
    pcd_model = o3d.geometry.PointCloud()
    pcd_model.points = o3d.utility.Vector3dVector(model)
    pcd_model.paint_uniform_color([0.7, 0.7, 0.7])

    pcd_shift = o3d.geometry.PointCloud()
    pcd_shift.points = o3d.utility.Vector3dVector(sample)
    pcd_shift.paint_uniform_color([0, 1, 0])

    pcd_flow = o3d.geometry.PointCloud()
    pcd_flow.points = o3d.utility.Vector3dVector(model+flow)
    pcd_flow.paint_uniform_color([1, 0, 0])

    print(len(pcd_model.points))

    o3d.visualization.draw_geometries([pcd_model,pcd_shift,pcd_flow])

def vizualize2(sample,flow):
    """
    draw in gray the model, in green the shifted model, in red the result obtained after the flow.
    """

    pcd_shift = o3d.geometry.PointCloud()
    pcd_shift.points = o3d.utility.Vector3dVector(sample)
    pcd_shift.paint_uniform_color([0, 1, 0])

    pcd_flow = o3d.geometry.PointCloud()
    pcd_flow.points = o3d.utility.Vector3dVector(sample+flow)
    pcd_flow.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd_shift,pcd_flow])

def vizualizeLabels(sample, labels):
    """
    draw in gray the model, in green the shifted model, in red the result obtained after the flow.
    """

    pcd_shift = o3d.geometry.PointCloud()
    pcd_shift.points = o3d.utility.Vector3dVector(sample)
    # labels = labels.reshape((labels.shape[0], 1))
    colors = np.zeros((labels.shape[0],3))
    idx_base  = np.where(labels==0)[0]
    idx_shift = np.where(labels==1)[0]
    idx_rotat = np.where(labels==2)[0]

    colors[idx_base,:] = np.tile(np.array([0.7,0.7,0.7]),(len(idx_base),1))
    colors[idx_shift,:] = np.tile(np.array([0,1,0]),(len(idx_shift),1))
    colors[idx_rotat,:] = np.tile(np.array([1,0,0]),(len(idx_rotat),1))
    pcd_shift.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_shift])

def vizualize_error(sample, labels,labels_pred):
    """
    draw in gray the model, in green the shifted model, in red the result obtained after the flow.
    """

    pcd_shift = o3d.geometry.PointCloud()
    pcd_shift.points = o3d.utility.Vector3dVector(sample)
    # labels = labels.reshape((labels.shape[0], 1))
    colors = np.zeros((labels.shape[0],3))
    idx_good  = np.where(labels==labels_pred)[0]
    idx_bad = np.where(labels!=labels_pred)[0]

    colors[idx_good,:] = np.tile(np.array([0.7,0.7,0.7]),(len(idx_good),1))
    colors[idx_bad,:] = np.tile(np.array([1,0,0]),(len(idx_good),1))
    pcd_shift.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd_shift])

def vizualizeflow(xyz1,xyz2,flow):

    data = [
        go.Scatter3d(
            x=xyz1[:, 0],
            y=xyz1[:, 1],
            z=xyz1[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='black'
            )
        ),
        go.Scatter3d(
            x=xyz2[:, 0],
            y=xyz2[:, 1],
            z=xyz2[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='green'
            )
        ),
        go.Cone(
            x=xyz1[:, 0],
            y=xyz1[:, 1],
            z=xyz1[:, 2],
            u=flow[:, 0],
            v=flow[:, 1],
            w=flow[:, 2],
            sizemode="absolute",
            sizeref=2,
            anchor="tip"
            )
            
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor="LightSteelBlue",
    )

    iplot(fig)
    return 

def vizualizeflow2(xyz1,flow):

    data = [
        go.Scatter3d(
            x=xyz1[:, 0],
            y=xyz1[:, 1],
            z=xyz1[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color='black'
            )
        ),
        go.Cone(
            x=xyz1[:, 0],
            y=xyz1[:, 1],
            z=xyz1[:, 2],
            u=flow[:, 0],
            v=flow[:, 1],
            w=flow[:, 2],
            sizemode="absolute",
            sizeref=2,
            anchor="tip"
            )
            
    ]
    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor="LightSteelBlue",
    )

    iplot(fig)
    return 
if __name__ =='__main__':

    # TRAIN_DIR = os.path.join(DATA_DIR,'train')
    # dict_trans = {0: 'rien', 1:'translate', 2:'rotate'}
    # dir = os.listdir(TRAIN_DIR)
    # for i,file in enumerate(dir):
        
    #     label = file.split("_")[-2][0]
    #     model_name = file.split("_")[0]+'_0_0_.npy'

        
    #     print(dict_trans[int(label)],"---   ---",file)


    #     model = np.load(os.path.join(TRAIN_DIR,file))[:,:3]
    #     shift = np.load(os.path.join(TRAIN_DIR,model_name))[:,:3]
    #     # flow_shift = np.load(os.path.join(TRAIN_DIR,file))[:,3:6]
    #     vizualize_sans_flow(model,shift)
    #     # vizualizeflow(model,shift,flow_shift)

    TRAIN_DIR = os.path.join(DATA_DIR,'seg/test')
    dict_trans = {0: 'rien', 1:'translate', 2:'rotate'}
    dir = os.listdir(TRAIN_DIR)
    print(TRAIN_DIR)
    for i,file in enumerate(dir):
        # if i != 1:
        #     continue
        label = file.split("_")[-2][0]
        model_name = file.split("_")[0]+'_0_0_.npy'
        print(dict_trans[int(label)],"---   ---",file)

        model = np.load(os.path.join(TRAIN_DIR,model_name))[:,:3]
        flow_shift = np.load(os.path.join(TRAIN_DIR,file))[:,3:6]
        shift =np.load(os.path.join(TRAIN_DIR,file))[:,:3] 
        
        
        # vizualizeflow2(shift,flow_shift)
        vizualize_sans_flow(model,shift)
        # vizualizeflow(model,shift,flow_shift)
        # vizualize2(shift,flow_shift)
        # vizualizeLabels(shift,np.load(os.path.join(TRAIN_DIR,file))[:,6])
        vizualize(model,shift,flow_shift)
        
    
    



