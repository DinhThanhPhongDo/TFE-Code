import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from time import time

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

def ransac(pts,pts_n, thresh_d,thresh_n,epoch=1000, tqdm_bool=False) :
    """
    points = np.array(N,3)
    pts_n  = np.array(N,3)
    """
    n_pts    = len(pts)
    epoch    = 500#int(np.log(1-0.95)/np.log(1-0.05**3))

    idx_pts  = np.arange(0,n_pts,1)
    best_inlier_mask  = None
    best_n_inliers = 3
    best_mse = np.inf

    iterator = range(epoch)
    if tqdm_bool:
        iterator = tqdm(iterator)

    for _ in iterator :
        # Step 1: Select 3 random points
        pts_sample = pts[np.random.choice(idx_pts,3,replace=False)]
    
        # Step 2: compute normals and distance to the origin of the candidate plane
        vecA = pts_sample[1, :] - pts_sample[0, :]
        vecB = pts_sample[2, :] - pts_sample[0, :]

        normal      =  np.cross(vecA, vecB)
        normal      =  normal/np.linalg.norm(normal)
        d           = -np.dot(normal,pts_sample[0,:])

        plane = np.array([normal[0], normal[1], normal[2], d])

        # Step 3: Voting process.
        dist_pt     = np.abs(np.dot(normal[:3],pts.T)+ d)
        inlier_mask = np.less_equal(dist_pt,thresh_d)*np.greater_equal(np.abs(np.dot(pts_n,normal)),thresh_n)
        n_inliers = np.sum(inlier_mask)

        # Step 4: keep in memory the best plane.
        mse = np.sqrt(np.sum(np.power(inlier_mask*dist_pt,2)))
    
        if n_inliers> best_n_inliers:
            
            best_mse          = mse
            best_plane        = plane
            best_inlier_mask  = inlier_mask
            best_n_inliers    = n_inliers
            best_centroid     = np.mean(pts[inlier_mask])
        elif n_inliers >= best_n_inliers and mse< best_mse:
            best_mse          = mse
            best_plane        = plane
            best_inlier_mask  = inlier_mask
            best_n_inliers    = n_inliers
            best_centroid     = np.mean(pts[inlier_mask])

    return best_plane,idx_pts[best_inlier_mask],best_centroid

                                
def planes_detection(xyz, voxel_size = 0.1, n_inliers = 500, n_plane=1, tqdm_bool=False, thresh_d=0.1,thresh_n=0.8,display=False):
    """
        xyz = np.array(N,3)
    """
    color = np.array([[255,128,0],
             [102,204,0],
             [0,204,204],
             [51,153,255],
             [255,0,255],
             [153,0,76],
             [255,255,153]])/255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # pcd_plane = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_plane = pcd
    pcd_plane.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

    planes       = []
    inliers_lst  = []
    i=0
    while(len(planes)<10 and len(pcd_plane.points) > 3) :
        pts_n        = np.asarray(pcd_plane.normals)
        pts          = np.asarray(pcd_plane.points)
        plane,inliers_idx, centroid = ransac(pts,pts_n, thresh_d=thresh_d,thresh_n=thresh_n,epoch=20, tqdm_bool=tqdm_bool)
        if display:
            print('remaining num points',len(pts))
            print('guessed num points',len(inliers_idx))
        if len(inliers_idx) < n_inliers :
            break

        inlier_cloud               = pcd_plane.select_by_index(inliers_idx)
        inlier_cloud, inliers_idx2 = inlier_cloud.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
        inlier_cloud.paint_uniform_color([0,1,0]) 
        inliers_idx = inliers_idx[inliers_idx2]

        outlier_cloud              = pcd_plane.select_by_index(inliers_idx, invert=True)
        outlier_cloud, outlier_idx = outlier_cloud.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
        outlier_cloud.paint_uniform_color([1,0,0]) 
        

        pcd_plane = outlier_cloud

        planes.append((plane,centroid))
        inliers_lst.append(inlier_cloud)

        if display:
            print('appended',len(inliers_idx))
            o3d.visualization.draw_geometries(inliers_lst+[outlier_cloud])


        i+=1
    
    
    return planes

def plane_matching(planes1,planes2) :

    rep = []

    for plane1,c1 in planes1:
        n1 = plane1[:3]
        best_align = 0

        for plane2,c2 in planes2:

            n2 = plane2[:3]

            if np.linalg.norm(c1-c2)< 0.5:
                if np.abs(np.dot(n1,n2))> best_align:

                    tr = transf_classify(n1,n2,c1,c2)
                    best_match = (plane1,plane2,c1,c2, tr)
                    best_align = np.abs(np.dot(plane1[:3],plane2[:3]))
        if best_align == 0:
            best_match = (plane1,None,c1,None, -1)
        rep.append(best_match)
    return rep

def transf_classify(n1,n2,c1,c2):  
    # parameters for the classification case: d <= 0.05 and a < 0.9998
    a = np.abs(np.dot(n1,n2))
    d = np.linalg.norm(c1-c2)
    if d >= 0.05 and a >0.999:
        return 1
    elif d < 0.05 and a> 0.999:
        return 0
    else:
        return 2

def change_detection(matches,xyz1,xyz2,display=False):
    color1 = np.array([[255,128,0],
             [102,204,0],
             [0,204,204],
             [51,153,255],
             [255,0,255],
             [153,0,76],
             [255,255,153]])/255
    color2 = np.array([[255,128,0],
                [102,204,0],
                [0,204,204],
                [51,153,255],
                [255,0,255],
                [153,0,76],
                [255,255,153]])/610
    dict_trans = {0: 'rien', 1:'translate', 2:'rotate'}
    dict_trans1 = {0: np.array([0.8,0.8,0.8]),
                  1: np.array([0,1,0]),
                  2: np.array([1,0,0])}
    dict_trans2 = {0: np.array([0.3,0.3,0.3]),
                  1: np.array([0,0.4,0]),
                  2: np.array([0.4,0,0])}
    pc1 = o3d.geometry.PointCloud()
    pc1.points = o3d.utility.Vector3dVector(xyz1)
    col1 = np.zeros((len(xyz1),3))

    pc1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    n1s        = np.asarray(pc1.normals)

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(xyz2)
    col2 = np.zeros((len(xyz2),3))
    pc2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    n2s        = np.asarray(pc2.normals)

    pred_label = np.zeros((len(xyz2)))

    it = 0
    for (plane1,plane2,c1,c2, tr) in matches:
        if hasattr(plane2, "__len__"):
            # print('alignment:',np.abs(np.dot(plane1[:3],plane2[:3])))
            # print('distance:',np.linalg.norm(c1-c2))
            # print('transform:',tr)
            thresh_d = 0.1
            thresh_n = 0.8
            # find inliers in pc1
            dist_pt1 = np.abs(np.dot(plane1[:3],xyz1.T)+ plane1[-1])
            inlier_mask1 = np.less_equal(dist_pt1,thresh_d)*np.greater_equal(np.abs(np.dot(n1s,plane1[:3])),thresh_n)

            dist_pt2 = np.abs(np.dot(plane2[:3],xyz2.T)+ plane2[-1])
            inlier_mask2 = np.less_equal(dist_pt2,thresh_d)*np.greater_equal(np.abs(np.dot(n2s,plane2[:3])),thresh_n)

            col1 += np.outer(inlier_mask1,dict_trans1[tr])
            col2 += np.outer(inlier_mask2,dict_trans2[tr])

            pred_label[inlier_mask2] = tr
            # col1 += np.outer(inlier_mask1,color1[it])
            # col2 += np.outer(inlier_mask2,color2[it])
        else:
            print('no matching')
        it += 1
    col1 = np.where(col1>1,1,col1)
    col2 = np.where(col2>1,1,col2)
    pc1.colors = o3d.utility.Vector3dVector(col1)
    pc2.colors = o3d.utility.Vector3dVector(col2)


    if display:
        o3d.visualization.draw_geometries([pc1,pc2])
    return pred_label


def model(source,target):

    planes1 = planes_detection(source,tqdm_bool=False)
    planes2 = planes_detection(target,tqdm_bool=False,display=False)
    matches = plane_matching(planes1,planes2)
    pred_label = change_detection(matches,source,target,display=False)

    return pred_label

def model2(base_dir,root_dir,data_dir,train_dir):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = base_dir
    data_dir = os.path.join(root_dir,'data')
    train_dir = os.path.join(data_dir,'cls/train')

    dict_trans = {0: 'rien', 1:'translate', 2:'rotate'}
    dir = os.listdir(train_dir)
    bad = 0
    bad_cls = 0
    total = 0
    tot_t = 0


    targets = []
    preds   = []
    for i,target_file in enumerate(dir):
        

        print("--------- iteration %d/%d --------"%(i,len(dir)))
        labels      = target_file.split("_")[-2][0]
        source_file = target_file.split("_")[0]+'_0_0_.npy'
        

        source = np.load(os.path.join(train_dir,source_file))[:,:3]
        target = np.load(os.path.join(train_dir,target_file))[:,:3]
        labels = np.load(os.path.join(train_dir,target_file))[:,-1]

        t0 = time()
        print('source:',source_file)
        planes1 = planes_detection(source,tqdm_bool=False)
        print('target:',target_file)
        planes2 = planes_detection(target,tqdm_bool=False,display=False)
        
        
        matches = plane_matching(planes1,planes2)
        pred_label = change_detection(matches,source,target,display=False)

        t1 = time()
        tot_t += t1-t0
        curr_bad   = np.sum(pred_label!=labels)
        curr_total = len(labels)

        print('number of false results:%d/%d (%5.2f)'%(curr_bad,curr_total, curr_bad/curr_total*100))
        print('time= %5.2f'%(t1-t0))
        bad += curr_bad
        total+=curr_total
        if pred_label[0] != labels[0]:
            bad_cls += 1
        

        targets.append(int(labels[0]))
        preds.append(int(pred_label[0]))

    # print('mAcc:',1-bad/total)
    # print('Bad classification: %d/%d (%5.2f)'%(bad_cls,len(dir),bad_cls/len(dir)*100))


    matrix = confusion_matrix(targets, preds)
    acc    = accuracy_score(targets,preds)
    prec   = precision_score(targets,preds,average=None)
    rec    = recall_score(targets,preds,average=None)
    f1     = f1_score(targets,preds,average=None)
    class_acc = matrix.diagonal()/matrix.sum(axis=1)

    print('avg acc %f     avg prec %f    avg recall %f    avg f1 %f'%(acc, precision_score(targets,preds,average='weighted'),recall_score(targets,preds,average='weighted'),f1_score(targets,preds,average='weighted')))
    print('accuracy',class_acc)
    print('precision:', prec)
    print('recall',rec)
    print('f1-score:',f1)
    print(matrix)








if __name__ =='__main__':
    model()

        



