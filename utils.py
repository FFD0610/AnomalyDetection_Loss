from cProfile import label
import os
import numpy as np
from tqdm import tqdm
import yaml
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import confusion_matrix 

#calculate the F1 for multi-classification
def each_class(labels,result,fault_flags):
    results = ''
    matrix = confusion_matrix(labels,result)
    batch_sum = matrix.flatten()
    batch_sum = np.sum(batch_sum)
    f1s,pres,recs,accs = [], [], [], []
    for fault_flag in fault_flags:
        index = fault_flags.index(fault_flag)
        TP = matrix[index,index]
        
        FP_TP = np.sum(matrix[:,index],axis=0)
        TP_FN = np.sum(matrix[index],axis=0)
        TN = batch_sum + TP - FP_TP - TP_FN 
        if FP_TP == 0:
            FP_TP = 1e10
        if TP_FN == 0:
            TP_FN = 1e10
        pre = float(TP)/FP_TP
        rec = float(TP)/TP_FN
        acc = float(TP+TN)/batch_sum
        if pre + rec == 0:
            F1 = 1/float(1e10) 
        else:
            F1 = 2*pre*rec/(pre+rec)

        f1s.append(F1)
        pres.append(pre)
        recs.append(rec)
        accs.append(acc)
        
        result = 'class {}: acc = {}, precision = {}, recall = {}, F1 score = {}\n'.format(fault_flag,acc,pre,rec,F1)

        results = results + result
    results = results + '\n'
    avg_F1 = sum(f1s) / len(f1s)
    pre = sum(pres) / len(pres)
    rec = sum(recs)/ len(recs)

    return results, matrix, avg_F1, pre, rec

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment

    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.find('_') if '_' in x else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')



'''
def plot(filedir,fault_flags,img_dir):
    params = {'xtick.labelsize':20,
              'ytick.labelsize':20,
              'legend.fontsize':20}
    plt.rcParams.update(params)
    for model_para in os.listdir(filedir):
        time_steps = int(model_para.split()[1])
        val_idxs = os.listdir(os.path.join(filedir,model_para))
        for val_idx in val_idxs:
            #slding result
            pool = layers.AveragePooling1D(pool_size=time_steps)
            fig_dir = os.path.join(img_dir,model_para,'{} imgs'.format(val_idx))
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            for fault_flag in fault_flags:
                if fault_flag == 'Normal':
                    continue
                else:
                    file_dir = os.path.join(filedir,model_para,val_idx,'slide_batch_eval',fault_flag)
                    r_ranges = os.listdir(file_dir)
                    for r_range in r_ranges:
                        np_dir = os.path.join(file_dir,r_range)
                        data = np.load(os.path.join(np_dir,'softmax.npy'))
                        result = pool(data)
                        fig = plt.figure(figsize=(15,9), dpi=300)

                        ax = fig.add_subplot(111)
                        for i in range(result.shape[-1]):
                            x = result[:,:,i]
                            #put the belonged class and normal class in bold
                            if i == fault_flags.index(fault_flag):
                                ax.plot(x[500:2000],label=fault_flags[i], linewidth='10')
                            elif i == (result.shape[-1]-1):
                                ax.plot(x[500:2000],label=fault_flags[i], linewidth='10')
                            else:
                                ax.plot(x[500:2000],label=fault_flags[i], linewidth='5')

                        #auxiliary line
                        plt.axvline(x=1000-int(time_steps)-500,c='black',linestyle='--', linewidth='3')
                        plt.axvline(x=1000-500,c='black',linestyle='--', linewidth='3')
                        plt.xlabel('time(sec))', fontsize=25)   
                        #plt.xticks(np.arange(0, 3001, 500),np.arange(int(time_steps),30001+int(time_steps),500))
                        
                        plt.ylabel("classification score", fontsize=25)         # Y轴标签
                        plt.legend()
                        #ax.lengend()
                        #plt.rc('lengend',fontsize=50)
                        plt.gcf().autofmt_xdate()
                        title_str = '{}_{}_{}'.format(fault_flag,r_range,time_steps) 
                        plt.title(title_str,fontsize=25, color='black', pad=20)
                        plt.savefig(os.path.join(fig_dir,title_str+'.png'))                             
'''

if __name__ == '__main__':
    filedir = '/mnt/data/src/0304_sl_accord/log1/log/1' #/att 50
    fault_flags = ['12_FLW4_1', '29_VLVBPS', '30_VLVACL', '31_GCI', '32_34_GC1P1_3', '35_GC2P', '38_WPMP1', '41_VVPWC1', '42_VVPWC2']
    img_dir = '/mnt/data/src/0304_sl_accord/log1/log/NED/sliding'
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir)
    plot(filedir,fault_flags,img_dir)
