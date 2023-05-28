from __future__ import print_function
import os
import shutil
import torch
import torch.nn as nn
from dataloaders.FlowNetDataloader import Object3DDataset
from models.FlowNet3D import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm    

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    


def main(input_dir,output_dir):

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    eval_loader = DataLoader(Object3DDataset(input_dir,mode='eval',npoints=2048))

    net = FlowNet3D(None).cuda()
    net.apply(weights_init)

    model_path = os.path.join(ROOT_DIR,'pretrained_model/flownet/model.github.t7')
    net.load_state_dict(torch.load(model_path), strict=False)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if torch.cuda.device_count() > 1:
        print()
        net = nn.DataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    else:
        pass
        #raise Exception('Not implemented')

    net.eval()
    for i, data in tqdm(enumerate(eval_loader), total = len(eval_loader)):
        pc1, pc2, color1, color2, flow, mask, labels, sample_filename = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()

        flow_pred = net(pc2, pc1, color1, color2).permute(0,2,1)
        
        flow_pred = flow_pred.cpu().detach().numpy() #1xNx3
        pc2       = pc2.cpu().detach().numpy() #1x3xN
        pc1       = pc1.cpu().detach().numpy() #1x3xN
        flow_pred = flow_pred[0,:,:] #Nx3
        pc2       = pc2[0,:,:] #3xN
        pc1       = pc1[0,:,:] #3xN
        labels = labels.cpu().detach().numpy()[0,:]
        labels = np.reshape(labels,(len(labels),1))

        output1 = np.concatenate((pc2.T,flow_pred),axis=1)
        output = np.concatenate((output1,labels),axis=1)

        np.save(os.path.join(output_dir, sample_filename[0]),output)
    print('FINISH')

def delete_files(data_dir):
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = BASE_DIR
    DATA_DIR = os.path.join(ROOT_DIR,'data/')

    input_dirs = [os.path.join(DATA_DIR,'cls/train'),
                 os.path.join(DATA_DIR,'cls/test'),
                 os.path.join(DATA_DIR,'cls_noisy/train'),
                 os.path.join(DATA_DIR,'cls_noisy/test'),
                 os.path.join(DATA_DIR,'seg/train'),
                 os.path.join(DATA_DIR,'seg/test'),
                 os.path.join(DATA_DIR,'seg_noisy/train'),
                 os.path.join(DATA_DIR,'seg_noisy/test')]
    
    output_dirs = [os.path.join(DATA_DIR,'cls_flow/train'),
                 os.path.join(DATA_DIR,'cls_flow/test'),
                 os.path.join(DATA_DIR,'cls_flow_noisy/train'),
                 os.path.join(DATA_DIR,'cls_flow_noisy/test'),
                 os.path.join(DATA_DIR,'seg_flow/train'),
                 os.path.join(DATA_DIR,'seg_flow/test'),
                 os.path.join(DATA_DIR,'seg_flow_noisy/train'),
                 os.path.join(DATA_DIR,'seg_flow_noisy/test')]
    for i in range(len(input_dirs)):

        print('input=%s \noutput=%s'%(input_dirs[i],output_dirs[i]))
        delete_files(output_dirs[i])
        main(input_dirs[i],output_dirs[i])