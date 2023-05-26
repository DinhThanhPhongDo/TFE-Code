from __future__ import print_function
import os
import torch
import torch.nn as nn
from data_util.myFlowDataloader import Object3DDataset
from models import FlowNet3D
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm    
from My_visualization import vizualize_sans_flow,vizualizeflow,vizualize

def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
DATAFLOW_DIR = os.path.join(ROOT_DIR,'data/cls/-')

def main():

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # CUDA settings
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)

    # test_loader = DataLoader(BlenderDataset(20000))
    partition = "train"
    test_loader = DataLoader(Object3DDataset(partition))

    model_path = os.path.join(ROOT_DIR,'pretrained_model/model.best.t7')
    net = FlowNet3D(None).cuda()
    net.apply(weights_init)
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
    for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
        pc1, pc2, color1, color2, labels, sample_filename = data
        pc1 = pc1.cuda().transpose(2,1).contiguous()
        pc2 = pc2.cuda().transpose(2,1).contiguous()
        color1 = color1.cuda().transpose(2,1).contiguous()
        color2 = color2.cuda().transpose(2,1).contiguous()

        flow_pred = net(pc1, pc2, color1, color2).permute(0,2,1)
        
        flow_pred = flow_pred.cpu().detach().numpy() #1xNx3
        pc2       = pc2.cpu().detach().numpy() #1x3xN
        pc1       = pc1.cpu().detach().numpy() #1x3xN
        flow_pred = flow_pred[0,:,:] #Nx3
        pc2       = pc2[0,:,:] #3xN
        pc1       = pc1[0,:,:] #3xN
        labels = labels.cpu().detach().numpy()[0,:,:]
        # print(flow_pred.shape)
        # print(pc2.shape)
        output1 = np.concatenate((pc2.T,flow_pred),axis=1)
        output = np.concatenate((output1,labels),axis=1)


        # dict_trans = {0: 'rien', 1:'translate', 2:'rotate'}
        # label = sample_filename[0].split("_")[-2][0]
        # print(dict_trans[int(label)],'--- ---',sample_filename[0] )
        # vizualize_sans_flow(pc1.T,pc2.T)
        # vizualize(pc1.T,pc2.T,flow_pred)
        # vizualizeflow(pc1.T,pc2.T,flow_pred)

        # vizualizeflow(pc1.T,pc2.T,flow_pred)
        # strangely, sample_filename is a tuple?
        np.save(os.path.join(DATAFLOW_DIR, partition, sample_filename[0]),output)
    print('FINISH')

if __name__ == '__main__':
    main()