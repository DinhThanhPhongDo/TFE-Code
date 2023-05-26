from multiprocessing import freeze_support
import numpy as np
import torch
import os
import importlib
from tqdm import tqdm
import models.provider as provider
import sys

from dataloaders.PlaneDataLoader import PlaneDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/cls/')

def test(model, loader, num_class=40):
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

def main() : 
    train_dataset = PlaneDataLoader(root=DATA_DIR+"train")
    test_dataset  = PlaneDataLoader(root=DATA_DIR+"test")
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1, drop_last=True)
    testDataLoader  = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

    num_class = 3

    sys.path.append(os.path.join(BASE_DIR, 'models'))
    model = importlib.import_module('pointnet2_cls_ssg')

    classifier = model.get_model(num_class, normal_channel=True)
    criterion = model.get_loss()
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True
    classifier.apply(inplace_relu)

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    if 'Adam' == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-4
            )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    print('try loading pretrained model...')
    try:
        checkpoint = torch.load(os.path.join(BASE_DIR, 'pretrained_model/pointnet_cls/best_cls_no_noise.pth'))
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('pretrained model loaded!')
    except Exception:
        print('could not load pretrained model... start from scratch')
    print('Start training...')
    start_epoch = 0
    for epoch in range(start_epoch, 20):
        #log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 200))
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 20))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        #log_string('Train Instance Accuracy: %f' % train_instance_acc)
        print('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            #log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            #log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            if (instance_acc >= best_instance_acc):
                #logger.info('Save model...')
                savepath = str("pretrained_model/pointnet_cls") + '/current_model.pth'
                #log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                print('saving...')
                torch.save(state, savepath)
            global_epoch += 1

    #logger.info('End of training...')

if __name__ == '__main__':
    freeze_support()
    main()