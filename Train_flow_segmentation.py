import numpy as np
import torch
import os
import importlib
from tqdm import tqdm
import models.provider as provider
import sys
import time

from dataloaders.SegDataLoader import SegDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/seg/')

classes = ['base', 'translation', 'rotation']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def main() : 
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    NUM_CLASSES = 3
    NUM_POINT = 4096
    BATCH_SIZE = 8
    train_dataset = SegDataLoader(data_root=DATA_DIR+"train",num_point = NUM_POINT, block_size=6)
    test_dataset = SegDataLoader(data_root=DATA_DIR+"test",num_point = NUM_POINT, block_size=6)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
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

    best_iou = 0
    if True:
        checkpoint = torch.load(os.path.join(BASE_DIR, 'pretrained_model/pointnet_seg/acc88.pth'))
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

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = 100

    global_epoch = 0
    

    '''TRANING'''
    for epoch in range(start_epoch, 20):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, 20))
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
        classifier = classifier.train()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            # points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:9] = provider.rotate_point_cloud_with_normal_9(points[:,:,0:9])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        print('Training mean loss: %f' % (loss_sum / num_batches))
        print('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            print('Save model...')
            savepath = str("pretrained_model/pointnet_seg/") + 'backup_model.pth'
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
                # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
                points[:, :, 0:9] = provider.rotate_point_cloud_with_normal_9(points[:,:,:])
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
            print(labelweights)
            print(predlabelweights)
            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            print('eval mean loss: %f' % (loss_sum / float(num_batches)))
            print('eval point avg class IoU: %f' % (mIoU))
            print('eval point accuracy: %f' % (total_correct / float(total_seen)))
            print('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

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
                savepath = str("pretrained_model/pointnet_seg") + '/current_model.pth'
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

if __name__ == '__main__':
    #freeze_support()
    main()