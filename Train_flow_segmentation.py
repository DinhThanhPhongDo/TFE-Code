import numpy as np
import torch
import torch.nn as nn
import os
import pickle
import importlib
from tqdm import tqdm
#import matplotlib.pyplot as plt
import models.provider as provider
import sys
import time
import torch.nn as nn

from dataloaders.SegDataLoader import SegDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATA_DIR = os.path.join(ROOT_DIR,'data/')

classes = ['base', 'translation', 'rotation']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def main(data_dir,filename,n_epoch = 10) : 
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    NUM_CLASSES = 3
    NUM_POINT = 4096
    BATCH_SIZE = 8
    train_dataset = SegDataLoader(data_root=data_dir+"train",num_point = NUM_POINT, block_size=3)
    test_dataset = SegDataLoader(data_root=data_dir+"test",num_point = NUM_POINT, block_size=3)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                                  pin_memory=True, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(train_dataset.labelweights).cuda()
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    model = importlib.import_module('pointnet2_sem_seg_msg')

    classifier = model.get_model(NUM_CLASSES)
    criterion = model.get_loss()
    def inplace_relu(m):
        classname = m.__class__.__name__
        if classname.find('ReLU') != -1:
            m.inplace=True
    classifier.apply(inplace_relu)

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    if torch.cuda.device_count() > 1:
        print()
        classifier = nn.DataParallel(classifier)
        criterion  = nn.DataParallel(criterion)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    best_iou = 0
    if True:
        checkpoint = torch.load(os.path.join(BASE_DIR, 'pretrained_model/pointnet_seg/current_model.pth'))
        start_epoch = checkpoint['epoch']
        start_epoch = 0
        if 'class_avg_iou' in checkpoint.keys() :
            best_iou = checkpoint['class_avg_iou']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print('Use pretrain model')
    else :
        start_epoch = 0
        classifier = classifier.apply(weights_init)
        print('model from scratch')
    
    if 'Adam' == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=0.001,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=1e-4
            )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    if torch.cuda.device_count() > 1:
        classifier = nn.DataParallel(classifier)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 100

    global_epoch = 0
    
    train_IoU = np.zeros((n_epoch,4))
    train_Acc = np.zeros((n_epoch,4))
    test_IoU = np.zeros((n_epoch,4))
    test_Acc = np.zeros((n_epoch,4))
    '''TRANING'''
    for epoch in range(start_epoch, n_epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, n_epoch))
        lr = max(0.001 * (1e-4 ** (epoch // 10)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0

        total_seen_class     = [0 for _ in range(NUM_CLASSES)]
        total_correct_class  = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            t0 = time.time()
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:9] = provider.rotate_point_cloud_with_normal_9(points[:,:,0:9])
            points[:, :, 0:9], target, _ = provider.shuffle_data(points[:,:,0:9],target)

            points = torch.Tensor(points)
            
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            t1 = time.time()
            seg_pred, trans_feat = classifier(points)
            t2 = time.time()
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            
            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            
            target      = target.view(-1, 1)[:, 0]
            t3 = time.time()
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
            
            for l in range(NUM_CLASSES):
                total_seen_class[l] += np.sum((batch_label == l))
                total_correct_class[l]  += np.sum((pred_choice == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))
            
            # print('\n')
            # print('\n',t1-t0,t2-t1,t3-t2)
            # print(t3-t0)
        
        for l in range(NUM_CLASSES):
            
            train_IoU[epoch,l] = total_correct_class[l] / (total_iou_deno_class[l]+ 1e-6)
            train_Acc[epoch,l] = total_correct_class[l] / (total_seen_class[l]+1e-6)

        print('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 1 == 0:
            print('Save model...')
            savepath = str("pretrained_model/pointnet_seg/") + 'current_model.pth'
            #log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        
        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            predlabelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            print('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                
                points = points.data.numpy()
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points[:, :, 0:9] = provider.rotate_point_cloud_with_normal_9(points[:,:,:])
                points[:, :, 0:9], target, _ = provider.shuffle_data(points[:,:,0:9],target)
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp
                tmp, _ = np.histogram(pred_val, range(NUM_CLASSES + 1))
                predlabelweights += tmp
                

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            # print(labelweights)
            # print(predlabelweights)
            for l in range(NUM_CLASSES):
                
                test_IoU[epoch,l] = total_correct_class[l] / (total_iou_deno_class[l]+ 1e-6)
                test_Acc[epoch,l] = total_correct_class[l] / (total_seen_class[l]+1e-6)

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6))
            print('eval mean loss: %f' % (loss_sum / float(num_batches)))
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
            print('Eval mean loss: %f' % (loss_sum / num_batches))
            print('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                #logger.info('Save model...')
                savepath = str("pretrained_model/pointnet_seg/") + filename+'.pth'
                #log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                #log_string('Saving model....')
            print('Best mIoU: %f' % best_iou)
        global_epoch += 1
    #logger.info('End of training...')
    train_IoU[:,3] = np.mean(train_IoU[:,:3],axis=1)
    train_Acc[:,3] = np.mean(train_Acc[:,:3],axis=1)
    test_IoU[:,3] = np.mean(test_IoU[:,:3],axis=1)
    test_Acc[:,3] = np.mean(test_Acc[:,:3],axis=1)
    result_dict = {
        "train Acc"      : train_Acc,
        "train IoU"      : train_IoU,

        "test Acc"       : test_Acc,
        "test IoU"       : test_IoU,

    }
    with open(os.path.join(ROOT_DIR,'logs/'+filename+'.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
    return 





if __name__ == '__main__':
    # freeze_support()

    name = [#'seg_nonoise',
            'seg_noise',
            #'seg_flow_nonoise',
            'seg_flow_noise']
    data_dirs = [#os.path.join(DATA_DIR,'seg/'),
                os.path.join(DATA_DIR,'seg_noisy/'),
                #os.path.join(DATA_DIR,'seg_flow/'),
                os.path.join(DATA_DIR,'seg_flow_noisy/')]
    for i in range(len(data_dirs)):
        print('\n \n***  training model:%s ***'%name[i])
        main(data_dirs[i],name[i],30)
        

