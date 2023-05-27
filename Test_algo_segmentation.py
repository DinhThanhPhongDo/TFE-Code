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
    train_dir = os.path.join(data_dir,'seg/test')

    dir = os.listdir(train_dir)
    tot_t = 0

    seg_label_to_cat = {0:'no transform',1:'translation',2:'rotation'}
    NUM_CLASSES = 3
    num_batches = len(dir)
    total_correct = 0
    total_seen = 0
    labelweights = np.zeros(3)
    predlabelweights = np.zeros(3)
    total_seen_class      = [0 for _ in range(3)]
    total_correct_class   = [0 for _ in range(3)]
    total_iou_deno_class  = [0 for _ in range(3)]

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

        batch_label = labels
        pred_val = pred_label

        correct = np.sum((pred_label == labels))
        total_correct += correct
        total_seen += len(pred_label)
        tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
        labelweights += tmp
        tmp, _ = np.histogram(pred_val, range(NUM_CLASSES + 1))
        predlabelweights += tmp

        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum((batch_label == l))
            total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
            total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

    labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
    IoU  = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
    mIoU = np.mean(IoU)

    print('eval point avg class IoU: %f' % (mIoU))
    print('eval point accuracy: %f' % (total_correct / float(total_seen)))
    print('eval point avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
    
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
            total_correct_class[l] / float(total_iou_deno_class[l]))

    print(iou_per_class_str)
    print('Eval accuracy: %f' % (total_correct / float(total_seen)))

if __name__ == '__main__':
    test_classification()