from multiprocessing import freeze_support
import numpy as np
import torch
import os
import importlib
from tqdm import tqdm
import sys

from dataloaders.PlaneDataLoader import PlaneDataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/cls/')

def test(model, loader, num_class=3):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return instance_acc, class_acc

def test2(model, loader, num_class=3):

    classifier = model.eval()
    targets = []
    preds   = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1)

        pred, _ = classifier(points)

        targets.extend(target.cpu().numpy())
        preds.extend(pred.data.max(1)[1].cpu().numpy())
    
    print(targets)
    print(preds)
    
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


    


    

def main() : 
    test_dataset  = PlaneDataLoader(root=DATA_DIR+"test")
    testDataLoader  = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

    num_class = 3

    sys.path.append(os.path.join(BASE_DIR, 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')

    classifier = model.get_model(num_class, normal_channel=True)
    classifier = classifier.cuda()

    print('try loading pretrained model...')
    try:
        checkpoint = torch.load(os.path.join(BASE_DIR, 'pretrained_model/pointnet_cls/current_model.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('pretrained model loaded!')
    except Exception:
        print('could not load pretrained model... start from scratch')
    with torch.no_grad():
        test2(classifier.eval(), testDataLoader, num_class=num_class)


    

if __name__ == '__main__':
    freeze_support()
    main()