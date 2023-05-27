from models.change_detection import model
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

if __name__ == '__main__':
    test_classification()