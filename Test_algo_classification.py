from models.change_detection import model
from texttable import Texttable
import latextable
from time import time
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

def test_classification():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = base_dir
    data_dir = os.path.join(root_dir,'data')
    train_dir = os.path.join(data_dir,'cls/train')

    dir = os.listdir(train_dir)
    tot_t = 0


    targets = []
    preds   = []

    for i,target_file in tqdm(enumerate(dir), total=len(dir)):
        labels      = target_file.split("_")[-2][0]
        source_file = target_file.split("_")[0]+'_0_0_.npy'
        

        source = np.load(os.path.join(train_dir,source_file))[:,:3]
        target = np.load(os.path.join(train_dir,target_file))[:,:3]
        labels = np.load(os.path.join(train_dir,target_file))[:,-1]

        t0 = time()
        pred_label = model(source,target)
        t1 = time()
        tot_t += t1-t0
        

        targets.append(int(labels[0]))
        preds.append(int(pred_label[0]))


    matrix = confusion_matrix(targets, preds)
    class_acc = matrix.diagonal()/matrix.sum(axis=1)
    acc    = accuracy_score(targets,preds)
    prec   = precision_score(targets,preds,average=None)
    prec_avg = precision_score(targets,preds,average='weighted')
    rec    = recall_score(targets,preds,average=None)
    rec_avg = recall_score(targets,preds,average='weighted')
    f1     = f1_score(targets,preds,average=None)
    f1_avg = f1_score(targets,preds,average='weighted')
    matrix = confusion_matrix(targets, preds,normalize='true')

    print(matrix)
    table = Texttable()
    table.set_cols_align(["l", "c", "c", "c", "c"])
    print("%.3f"%class_acc[1])
    table.add_rows([["Evaluation Metric", "No transformation", "Translation","Rotation", "Average"],
                    ["Accuracy", "%.3f"%class_acc[0], "%.3f"%class_acc[1], "%.3f"%class_acc[2], "%.3f"%acc],
                    ["Precision",     "%.3f"%prec[0],      "%.3f"%prec[1],      "%.3f"%prec[2], "%.3f"%prec_avg],
                    ["Recall",         "%.3f"%rec[0],       "%.3f"%rec[1],       "%.3f"%rec[2], "%.3f"%rec_avg],
                    ["f1-Score",        "%.3f"%f1[0],        "%.3f"%f1[1],        "%.3f"%f1[2], "%.3f"%f1_avg]])
    print(table.draw())
    print(latextable.draw_latex(table, caption="Classification Result", label="tab:cls_res"))

    table1 =Texttable()
    table1.set_cols_align(["l", "c", "c","c"])
    table1.add_rows([["","No transformation","Translation","Rotation"],
                    ["Predicted No transformation", "%.3f"%matrix[0,0], "%.3f"%matrix[0,1], "%.3f"%matrix[0,2]],
                    ["Predicted Translation"      , "%.3f"%matrix[1,0], "%.3f"%matrix[1,1], "%.3f"%matrix[1,2]],
                    ["Predicted Rotation"         , "%.3f"%matrix[2,0], "%.3f"%matrix[2,1], "%.3f"%matrix[2,2]]])
    print(table1.draw())
    print(latextable.draw_latex(table1, caption="Classification Result", label="tab:cls_res"))
1
if __name__ == '__main__':
    test_classification()
    # print( '%.2f'%1)