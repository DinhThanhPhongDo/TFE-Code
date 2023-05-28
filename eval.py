    

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/cls_noisy/')


with open(os.path.join(ROOT_DIR,'logs/seg_flow_noise.pkl'), 'rb') as f:
        result = pickle.load(f)
    
y1 = result["train Acc"]
y2 = result["train IoU"]
x = np.arange(1,len(y1)+1,1)
plt.plot(x,y1[:,0],label='no')
plt.plot(x,y1[:,1],label='translation')
plt.plot(x,y1[:,2],label='rotation')
plt.plot(x,y1[:,3],label='average')
plt.legend()
plt.title('Acc')
plt.xlabel('epoch')
plt.ylabel('Acc')
plt.grid()
plt.show()