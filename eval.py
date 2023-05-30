    

import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import matplotlib.lines as mlines
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR# os.path.abspath(os.path.join(BASE_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR,'data/cls_noisy/')


def eval_seg(filename):
        def seg_plot(x,y1,y_label,plotname,display=False):
                fig, ax = plt.subplots()
                # Plot the lines with different colors and linestyles
                ax.plot(x, y1[:,0],linewidth=1, color='black', linestyle='--',label='base')
                ax.plot(x, y1[:,1],linewidth=1, color='green', linestyle='--',label='translation')
                ax.plot(x, y1[:,2],linewidth=1, color='darkred', linestyle='--',label='rotation')
                ax.plot(x, y1[:,3],linewidth=2, color='goldenrod', linestyle='-',label='average')


                # Add legend and labels
                ax.legend()
                ax.set_xlabel('epochs')
                ax.set_ylabel(y_label)
                ax.grid()

                # Display the plot
                plt.savefig(os.path.join(ROOT_DIR,'plots/'+plotname))
                if display:
                        plt.show()
                plt.close()

        with open(os.path.join(ROOT_DIR,'logs/'+filename+'.pkl'), 'rb') as f:
                result = pickle.load(f)

        train_Acc = result['train Acc']
        train_IoU = result['train IoU']
        test_Acc  = result['test Acc']
        test_IoU  = result['test IoU']
        x = np.arange(1,len(train_Acc)+1,1)

        seg_plot(x,train_Acc,'Accuracy',filename+'_train_acc')
        seg_plot(x,test_Acc,'Accuracy',filename+'_test_acc')

        seg_plot(x,train_IoU,'IoU',filename+'_train_iou')
        seg_plot(x,test_IoU,'IoU',filename+'_test_iou')



def eval_cls(filename):
        def cls_plot(x,y1,y2,y_label,plotname,display=False):
                fig, ax = plt.subplots()
                # Plot the lines with different colors and linestyles
                ax.plot(x, y1[:,0],linewidth=1, color='black', linestyle='--',label='base')
                ax.plot(x, y1[:,1],linewidth=1, color='green', linestyle='--',label='translation')
                ax.plot(x, y1[:,2],linewidth=1, color='darkred', linestyle='--',label='rotation')
                ax.plot(x, y2     ,linewidth=2, color='goldenrod', linestyle='-',label='average')


                # Add legend and labels
                ax.legend()
                ax.set_xlabel('epochs')
                ax.set_ylabel(y_label)
                ax.grid()

                # Display the plot
                plt.tight_layout()
                
                if display:
                        plt.show()
                plt.savefig(os.path.join(ROOT_DIR,'plots/'+plotname))
                plt.close()

        with open(os.path.join(ROOT_DIR,'logs/'+filename+'.pkl'), 'rb') as f:
                result = pickle.load(f)

        train_class_accs        = result['train class accuracy']
        train_accs              = result['train average accuracy']
        train_precs             = result['train class precision']
        train_prec_avgs         = result['train average precision']
        train_recs              = result['train class recall']
        train_rec_avgs          = result['train average recall']
        train_f1s               = result['train class f1-score']
        train_f1_avgs           = result['train average f1-score']

        eval_class_accs         = result['test class accuracy']
        eval_accs               = result['test average accuracy']
        eval_precs              = result['test class precision']
        eval_prec_avgs          = result['test average precision']
        eval_recs               = result['test class recall']
        eval_rec_avgs           = result['test average recall']
        eval_f1s                = result['test class f1-score']
        eval_f1_avgs            = result['test average f1-score']
        x = np.arange(1,len(train_accs)+1,1)

        cls_plot(x,train_class_accs,train_accs,'Accuracy',filename+'_train_acc')
        cls_plot(x,eval_class_accs,eval_accs,'Accuracy',filename+'_test_acc')

        cls_plot(x,train_precs,train_prec_avgs,'Precision',filename+'_train_prec')
        cls_plot(x,eval_precs,eval_prec_avgs,'Precision',filename+'_test_prec')

        cls_plot(x,train_recs,train_rec_avgs,'Recall',filename+'_train_rec')
        cls_plot(x,eval_recs,eval_rec_avgs,'Recall',filename+'_test_rec')

        cls_plot(x,train_f1s,train_f1_avgs,'F1-Score',filename+'_train_f1')
        cls_plot(x,eval_f1s,eval_f1_avgs,'F1-Score',filename+'_test_f1')





if __name__ == '__main__':
        # eval_cls('cls_nonoise')
        eval_seg('seg_nonoise')
        eval_seg('seg_noise')
        
        eval_seg('seg_flow_nonoise')
        eval_seg('seg_flow_noise')
