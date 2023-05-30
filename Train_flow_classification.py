from multiprocessing import freeze_support
import numpy as np
import torch
import os
import importlib
from tqdm import tqdm
import models.provider as provider
import sys
import pickle 
import matplotlib.pyplot as plt

from dataloaders.PlaneDataLoader import PlaneDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/')

def test(model, loader):

    classifier = model.eval()
    targets = []
    preds   = []

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.cuda(), target.cuda()
        points = points.transpose(2, 1)

        pred, _ = classifier(points)

        targets.extend(target.cpu().numpy())
        preds.extend(pred.data.max(1)[1].cpu().numpy())
    
    matrix      = confusion_matrix(targets, preds)
    class_acc   = matrix.diagonal()/matrix.sum(axis=1)
    acc         = accuracy_score(targets,preds)
    prec        = precision_score(targets,preds,average=None)
    prec_avg    = precision_score(targets,preds,average='weighted')
    rec         = recall_score(targets,preds,average=None)
    rec_avg     = recall_score(targets,preds,average='weighted')
    f1          = f1_score(targets,preds,average=None)
    f1_avg      = f1_score(targets,preds,average='weighted')
    matrix      = confusion_matrix(targets, preds,normalize='true')

    return class_acc,acc,prec,prec_avg,rec,rec_avg,f1,f1_avg

def main(data_dir,filename,n_epoch = 10) : 
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    train_dataset = PlaneDataLoader(root=data_dir+"train")
    test_dataset  = PlaneDataLoader(root=data_dir+"test")
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

    # print('try loading pretrained model...')
    # try:
    #     checkpoint = torch.load(os.path.join(BASE_DIR, 'pretrained_model/pointnet_cls/best_cls_no_noise.pth'))
    #     classifier.load_state_dict(checkpoint['model_state_dict'])
    #     print('pretrained model loaded!')
    # except Exception:
    #     print('could not load pretrained model... start from scratch')
    
    print('Start training...')
    start_epoch = 0
    

    # evaluation metrics for train set
    train_class_accs = np.zeros((n_epoch,3))
    train_accs       = np.zeros((n_epoch))
    train_precs      = np.zeros((n_epoch,3))
    train_prec_avgs  = np.zeros((n_epoch,))
    train_recs       = np.zeros((n_epoch,3))
    train_rec_avgs   = np.zeros((n_epoch,))
    train_f1s        = np.zeros((n_epoch,3))
    train_f1_avgs    = np.zeros((n_epoch,))

    # evaluation metrics for test set
    eval_class_accs = np.zeros((n_epoch,3))
    eval_accs       = np.zeros((n_epoch,))
    eval_precs      = np.zeros((n_epoch,3))
    eval_prec_avgs  = np.zeros((n_epoch,))
    eval_recs       = np.zeros((n_epoch,3))
    eval_rec_avgs   = np.zeros((n_epoch,))
    eval_f1s        = np.zeros((n_epoch,3))
    eval_f1_avgs    = np.zeros((n_epoch,))
    
    for epoch in range(start_epoch, n_epoch):

        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, n_epoch))
        mean_correct = []
        targets      = []
        preds        = []

        '''TRAINING'''
        classifier = classifier.train()
        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            # Data Augmentation
            points              = points.data.numpy()
            points              = provider.random_point_dropout(points,max_dropout_ratio=0.5)
            points[:, :, 0:3]   = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, 0:9]   = provider.rotate_point_cloud_with_normal_6(points[:,:,0:9])
            points              = torch.Tensor(points)
            points              = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            targets.extend(target.cpu().numpy())
            preds.extend(pred.data.max(1)[1].cpu().numpy())
            loss.backward()
            optimizer.step()
            global_step += 1
        # evaluation of training 1 epoch
        matrix                      = confusion_matrix(targets, preds)
        train_class_accs[epoch,:]   = matrix.diagonal()/matrix.sum(axis=1)
        train_accs[epoch]           = accuracy_score(targets,preds)
        train_precs[epoch,:]        = precision_score(targets,preds,average=None)
        train_prec_avgs[epoch]      = precision_score(targets,preds,average='weighted')
        train_recs[epoch,:]         = recall_score(targets,preds,average=None)
        train_rec_avgs[epoch]       = recall_score(targets,preds,average='weighted')
        train_f1s[epoch,:]          = f1_score(targets,preds,average=None)
        train_f1_avgs[epoch]        = f1_score(targets,preds,average='weighted')
        train_instance_acc          = np.mean(mean_correct)

        print('Train Instance Accuracy: %f' % train_instance_acc)

        '''TESTING'''
        with torch.no_grad():
            class_acc,instance_acc,prec,prec_avg,rec,rec_avg,f1,f1_avg = test(classifier.eval(), testDataLoader)

            # evaluation on the whole testing set
            eval_class_accs[epoch,:] = class_acc
            eval_accs[epoch]       = instance_acc
            eval_precs[epoch,:]    = prec
            eval_prec_avgs[epoch]  = prec_avg
            eval_recs[epoch,:]     = rec
            eval_rec_avgs[epoch] = rec_avg
            eval_f1s[epoch,:]      = f1
            eval_f1_avgs[epoch]    = f1_avg

            class_acc = np.mean(class_acc)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            print('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            print('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
            if (instance_acc >= best_instance_acc):
                print('save model...')
                #logger.info('Save model...')
                savepath = str("pretrained_model/pointnet_cls/") + filename+'.pth'
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

    # saving evaluation metrics
    result_dict = {
        "train class accuracy"      : train_class_accs,
        "train average accuracy"    : train_accs,
        "train class precision"     : train_precs,
        "train average precision"   : train_prec_avgs,
        "train class recall"        : train_recs,
        "train average recall"      : train_rec_avgs,
        "train class f1-score"      : train_f1s,
        "train average f1-score"    : train_f1_avgs,

        "test class accuracy"       : eval_class_accs,
        "test average accuracy"     : eval_accs,
        "test class precision"      : eval_precs,
        "test average precision"    : eval_prec_avgs,
        "test class recall"         : eval_recs,
        "test average recall"       : eval_rec_avgs,
        "test class f1-score"       : eval_f1s,
        "test average f1-score"     : eval_f1_avgs,

    }
    with open(os.path.join(ROOT_DIR,'logs/'+filename+'.pkl'), 'wb') as f:
        pickle.dump(result_dict, f)
    return 


if __name__ == '__main__':
    # freeze_support()

    name = ['cls_nonoise','cls_noise','cls_flow_nonoise','cls_flow_noise']
    data_dirs = [os.path.join(DATA_DIR,'cls/'),
                os.path.join(DATA_DIR,'cls_noisy/'),
                os.path.join(DATA_DIR,'cls_flow/'),
                os.path.join(DATA_DIR,'cls_flow_noisy/')]
    for i in range(len(data_dirs)):
        if i == 0:
            continue
        print('\n \n***  training model:%s ***'%name[i])
        main(data_dirs[i],name[i],50)

    # with open('cls_noise.pkl', 'rb') as f:
    #     result = pickle.load(f)
    
    # y1 = result["train class accuracy"]
    # y2 = result["train average accuracy"]
    # x = np.arange(1,len(y1)+1,1)
    # plt.plot(x,y1[:,0],label='no')
    # plt.plot(x,y1[:,1],label='translation')
    # plt.plot(x,y1[:,2],label='rotation')
    # plt.plot(x,y2,label='average')
    # plt.legend()
    # plt.title('Acc')
    # plt.xlabel('epoch')
    # plt.ylabel('Acc')
    # plt.grid()
    # plt.show()