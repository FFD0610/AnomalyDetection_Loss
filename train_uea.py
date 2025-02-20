import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from uea_datasets import load_UEA
import time
from utils import Params #, each_class
import gc 
from losses.center_loss import CenterLoss
from losses.s_contrastive import  SupConLoss
from losses.mixed_loss import Mixed_calculation
from losses.asoftmax import SphereFace
from losses.soft_contrastive import SoftContra, HardContra 
from losses.auge import Augement
from losses.designed_loss import Design_loss
from networks import *
import shutil
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix 
'''
updated:
1. intro models to verify the loss function ( dlinear + timesnet)
2. softmax loss revision (check why )
3. se-based model cannot work for constrative loss based trainin process -> ensure why
4. center loss not best - revise this one 

'''

#revised from version train0206
'''
deal with the situation some samples with label but other without 
the dataset with label should be minimum to the ratio to the number without labels
so the dataset set with label and without label should be settled before input into the model training process 

'''
def each_class(labels,result,fault_flags):
    results = ''
    matrix = confusion_matrix(labels,result)
    batch_sum = matrix.flatten()
    batch_sum = np.sum(batch_sum)
    f1s,pres,recs,accs = [], [], [], []
    for fault_flag in fault_flags:
        index = fault_flag#
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


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


class TorchDataset(Dataset):
    def __init__(self, data, label, device, repeat=1, mask_ratio=0., mask_index=None):
        '''
        :param filelist: id for different file
        :param file_dir: divided file samples: saved_dir + time_stpes
        :param repeat: repeat times
        '''
        self.data = torch.from_numpy(data).to(torch.float).to(device=device)
        self.label = torch.from_numpy(label).to(torch.float).to(device=device)
        self.repeat = repeat
        self.len = self.__len__()
        self.mask_ratio = mask_ratio
        self.mask_label = self._mask_labels() if not mask_index else mask_index 


    def __getitem__(self, i):
        index = i % self.len #related postion
        label = self.label[i].clone().detach()
        data = self.data[i].clone().detach()

        #label =torch.tensor(self.label[i]) # all_data[:,-1] #channel dimension is in the first dimension
        #data = torch.tensor(self.data[i]) 
        masked_label = torch.tensor(float('nan'))*label if index in self.mask_label else label

        return data, label, masked_label

    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.data) * self.repeat
        return data_len


    def _mask_labels(self):
        """Randomly masks labels in the test set"""
        num_samples = self.len
        num_mask = int(num_samples * self.mask_ratio)
        if num_mask == 0:
            return {}
        mask_indices = torch.randperm(num_samples)[:num_mask].tolist()
        #mask_indices = {self.file_list[i]: float('nan') for i in mask_indices}  # Map to actual test indices

        return mask_indices


def load_datasets(train_data, train_labels, device, mask_ratio, mask_index):
    temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
    if temporal_missing[0] or temporal_missing[-1]:
        train_data = centerize_vary_length_series(train_data)
    train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
    #train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(train_labels).to(torch.float))
    train_dataset = TorchDataset(data=train_data,label=train_labels, device=device, mask_ratio=mask_ratio, mask_index=mask_index)
    #train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)#True)

    return train_loader

def get_args():
    parser = argparse.ArgumentParser(description='ACCORD data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/accord.yml', help='The path to the configuration file.')
    parser.add_argument('--save-dir', type=str, default='/mnt/database/torch_results/codetesting' , help='The directory of log files.')#./modeltestingsnr50/temtcnv3
    #/mnt/database/results_accord_torch/sekiguchi/timesnet1e5
    parser.add_argument('--name', default='', help='UEA dataset name of the task') #noiseintro080
    parser.add_argument('--epochs', type=int, default=200)################
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=10, help='K-fold cross validation.')

    return parser.parse_args()

def train(**kwargs):
    m = kwargs.get('m',None) #model_name
    epochs = kwargs.get('epochs', 1)
    channels = kwargs.get('channels',20)
    time_steps = kwargs.get('time_steps', 30)
    result_dir = kwargs.get('result_dir', None)
    train_loader = kwargs.get('train_loader',None)
    test_loader = kwargs.get('test_loader',None)
    fault_flags = kwargs.get('fault_flags',None)
    device = kwargs.get('device','cpu')
    #device = 'cpu'
    maxi = kwargs.get('maxi', None)
    mini = kwargs.get('mini', None)
    noise_intro = kwargs.get('noise_intro',False)
    loss_s_m  = kwargs.get('loss_s_m',None)
    loss_m = kwargs.get('loss_m',None)
    eval_pooling = kwargs.get('eval_pooling',False)
    pooling = kwargs.get('pooling', None)
    print(device)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    print(result_dir)
    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
  
    if m == 'seonly':
        model = SeCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)
        feat_dim = 128
    
    elif m == 'saonly':
        model = SaCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)

    elif m == 'dlinear':
        model = DLinear(time_steps=time_steps, channels=channels, classes = len(fault_flags) )
        feat_dim = 20
        if loss_m == 'mixed' or ( 'soft' or 'hard' ) in loss_m or 'aug' in loss_m :
            return 0, 0, 0, 0

    elif m == 'timesnet':
        model = TimesNet(time_steps=time_steps, channels=channels, classes = len(fault_flags) )
        feat_dim = 32

        if loss_m == 'mixed' or ( 'soft' or 'hard' ) in loss_m or 'aug' in loss_m:
            return 0, 0, 0, 0
    
    elif m == 'conv1d':
        model = CNN1D(time_steps=time_steps, channels=channels, classes = len(fault_flags))
        feat_dim  = 128

    elif 'mix' in m:
        if 'all' in m :
            model = MixedNet(time_steps=time_steps,channels=channels, classes = len(fault_flags ))
            feat_dim = 128+32+128
        if 'gru-se' in m:   
            model = MixedNet2(time_steps=time_steps,channels=channels, classes = len(fault_flags)  )
            feat_dim = 128+128
        if 'tcn-se' in m:
            model = MixedNet3(time_steps=time_steps,channels=channels, classes = len(fault_flags) )
            feat_dim = 128+32
        else: 
            model = TSEncoder(time_steps=time_steps,channels=channels, classes = len(fault_flags) )
            feat_dim = 128+32
    
    elif m == 'tcn':
        model = TCN(time_steps=time_steps,channels=channels, classes = len(fault_flags)  )
        feat_dim = 64

    model = model.to(device)
    #feat_dim = 128 if 'mix' not in m else feat_dim



    #loss definition
    if  'sphere' in loss_m:
        loss_cls = SphereFace(feat_dim, subj_num=len(fault_flags))
    
    loss_s = nn.CrossEntropyLoss().to(device)

    loss_c = None
    if 'center' in loss_m:
        loss_c = CenterLoss(feat_dim=feat_dim, classes=len(fault_flags))
        optimizer_center = optim.Adam(loss_c.parameters(), lr=0.001)

    elif loss_m == 'contrastive':
        loss_c = SupConLoss(contrast_mode='one')

    elif loss_m == 'supervised-contrastive':
        loss_c = SupConLoss(contrast_mode='all')

    elif loss_m == 'mixed':
        loss_c = Mixed_calculation()
    
    elif 'soft' in loss_m and 'softmax' not in loss_m:
        loss_c = SoftContra(soft_temporal=True, soft_instance=False)
    
    elif 'hard' in loss_m:
        loss_c = HardContra()
    
    elif 'design' in loss_m:
        loss_c = Design_loss(mask_ratio=0.05) #mask ratio for augementation of the dataset


    if 'aug' in loss_m:
        loss_aug = Augement(pooling=pooling)
    

         
    model = model.to(device)
    loss_s = loss_s.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    count = 0
    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        y_pred,y_true = None, None
        epoch_loss,epoch_accuracy = 0, 0
        labels_train,data_train = None, None

        
        train_start_time = time.time()
        for data, label, masked_label in tqdm(train_loader):
            #data is composed of the noise-intro and original part
            if noise_intro: 
                data = data[:,:channels]
            label = label.unsqueeze(1)
            masked_label = masked_label.unsqueeze(1)
                #('data size: ',data.size())
            #data = ((data- mini) / (maxi-mini)).float().to(device)
            #label = label.type(torch.LongTensor).to(device)
            #label = label.to(device)
            optimizer.zero_grad()

                
            if 'aug' in loss_m:
                crop_l, embed_aug , cls_aug, label_aug = loss_aug(data , label, model) #aumented output from the model
                #already pooling


            #all of the model have the two output: embedding features and the classification results
            if loss_c is not None:
                if 'center' in loss_m:
                    optimizer_center.zero_grad()
                    loss_c_score, diff, cls, label = loss_c(data=data, labels=label, model=model, pooling=pooling)
                    loss_c_score = loss_c_score if 'aug' not in loss_m else loss_c_score + loss_c(data=embed_aug, labels=label_aug, model=model, pooling=pooling)[0]
                elif 'design' in loss_m:
                    loss_c_score, diff, cls, label = loss_c(data=data, labels=masked_label, model=model, m=m)
                    #label = label.repeat(3,cls.size(-1)).type(torch.LongTensor).to(device)
                else:
                    loss_c_score, diff, cls, label = loss_c(data=data, labels=label, model=model) #only utilize the label for masked one 

            #features, cls = features.to(device), cls.to(device)
                #print(label.size(),cls.size())
                loss_s_score = loss_s(cls,label) if 'aug' not in loss_m  else loss_s(cls,label) + loss_s(cls_aug,label_aug)
                #loss = loss_c_score.clone() + loss_s_score.clone()
                loss = loss_c_score + loss_s_score
            else:
                #softmax in loss_m and might replaced by sphere face 

                embed, cls = model(data)

                #if 'aug' in loss_m:
                    #crop_l, embed_aug , cls_aug, label_aug = loss_aug(data , label, model)

                if pooling: 
                    cls = pooling(cls)
                    embed = pooling(embed)
                    label = pooling(label)
                label = label.repeat(1,cls.size(-1))
                label = label.type(torch.LongTensor).to(device)
                if 'sphere' in loss_m:
                    loss = loss_s(loss_cls(embed,label),label) if 'aug' not in loss_m  else loss_s(embed,label) + loss_s(embed_aug,label_aug)
                else:
                    loss = loss_s(cls,label) if 'aug' not in loss_m  else loss_s(cls,label) + loss_s(cls_aug,label_aug)
                
            if torch.isnan(loss).any():
                print(torch.isnan(data).any(), data, label, masked_label)
                print(torch.isnan(label).any(), torch.isnan(cls).any(), torch.isnan(loss_s_score), torch.isnan(loss_c_score))
            assert not torch.isnan(loss).any()
            loss.backward()
            optimizer.step()

            if 'center' in loss_m:
                
                loss_c.update_centers(diff)
                optimizer_center.step()



            data_train = cls.argmax(dim=1).flatten() if data_train is None else torch.cat([data_train, cls.argmax(dim=1).flatten()])
            labels_train = label.flatten() if labels_train is None else torch.cat([labels_train, label.flatten()])

            acc = (cls.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.detach().item() / len(train_loader)
            epoch_loss += loss.detach().item() / len(train_loader)
        gc.collect()

        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        labels_train = labels_train.cpu()
        data_train = data_train.cpu()
        print(labels_train.size(),data_train.size())
        _, matrix, train_F1, train_pre, train_rec = each_class(labels_train,data_train,fault_flags)

        each_cls_path = os.path.join(result_dir, 'train_each.txt')
        with open(each_cls_path, 'a+') as f:
            f.write('Epoch:{}\n'.format(epoch+1))
            f.write(str(_))
            f.write(str(matrix)+'\n')

        #evaluation 
        with torch.no_grad():
            model.eval()
            test_start_time = time.time()
            epoch_val_accuracy = 0
            epoch_val_loss = 0

            for data, label, masked_label in tqdm(test_loader):
                if noise_intro:
                    data = data[:,channels:]                    
                #data = ((data- mini) / (maxi-mini)).to(device).float()
                label = label.unsqueeze(1)
                masked_label = masked_label.unsqueeze(1)
                features, cls = model(data)
                features, cls = features.to(device), cls.to(device)
                label = label.repeat(1,cls.size(-1))
                if pooling:
                    cls = pooling(cls)
                    features = pooling(features)
                    label = pooling(label)

                label = label.type(torch.LongTensor).to(device)
                if 'sphere' in loss_m:
                    val_loss = loss_s(features, label)
                else:
                    val_loss = loss_s(cls, label)
                
                y_pred = cls.argmax(dim=1).flatten() if y_pred is None else torch.cat([y_pred, cls.argmax(dim=1).flatten()])
                y_true = label.flatten() if y_true is None else torch.cat([y_true, label.flatten()])
                assert len(y_true) == len(y_pred)
                
                acc = (cls.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc.detach().item() / len(test_loader)
                epoch_val_loss += val_loss.detach().item() / len(test_loader)
                
            test_end_time = time.time()
            testing_time = test_end_time - test_start_time

            gc.collect()

        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        _, matrix, avg_F1, avg_pre, avg_rec = each_class(y_true,y_pred,fault_flags)
        
        postfix = 'Epoch: {0}, loss: {1} Val F1: {2:.4f}, Val acc: {3:.4f}'.format(epoch+1, loss_m, avg_F1, epoch_val_accuracy)
        para = 'Epoch: {0} Train loss: {1:.4f} Train acc: {2:.4f} Train F1: {3:.4f} Val loss: {4:.4f} Val acc: {5:.4f}  \
                Val F1: {6:.4f}\n'.format(epoch, epoch_loss, epoch_accuracy, train_F1, epoch_val_loss, epoch_val_accuracy, avg_F1)

        print(postfix)
        print(_)

        each_cls_path = os.path.join(result_dir, 'loss_acc.txt')
        with open(each_cls_path, 'a+') as f:
            f.write(para)

        each_cls_path = os.path.join(result_dir, 'each.txt')
        with open(each_cls_path, 'a+') as f:
            f.write('Epoch:{}\n'.format(epoch+1))
            f.write(para)
            f.write(str(_))
            f.write(str(matrix)+'\n')
    
        time_path = os.path.join(result_dir, 'time.txt')
        with open(time_path, 'a+') as f:
            f.write('Epoch:{}\n'.format(epoch))
            f.write('training_time: {}\n'.format(training_time))
            f.write('testing_time: {}\n'.format(testing_time))
            f.write('train_start: {}\n'.format(train_start_time))
            f.write('train_end: {}\n'.format(train_end_time))
            f.write('test_start: {}\n'.format(test_start_time))
            f.write('test_end: {}\n'.format(test_end_time))

        print(
            f"Model : {m} Time_Steps: {time_steps} Epoch : {epoch+1} - loss : {epoch_loss:.4f} \
                - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_F1: {avg_F1:.4f}\n" #epoch_val_accuracy
        )

        if avg_F1 > best_f1:
            best_f1 = avg_F1
            best_pre = avg_pre
            best_rec = avg_rec
            best_acc = epoch_val_accuracy
            count = 0
        
            best_results_path = os.path.join(result_dir, 'best_results.txt')
            print('update the best results save in {}'.format(best_results_path))
            with open(best_results_path, 'w') as f:
                f.write('Epoch:{}\n'.format(epoch+1))
                f.write(para)
                f.write(str(_))
                f.write(str(matrix)+'\n')
            
            best_model_dir = os.path.join(result_dir, 'best_model')
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            
            best_model_path = os.path.join(best_model_dir, 'best_model{}'.format(str(epoch+1)))
            torch.save(model,best_model_path)
            best_model_path = os.path.join(best_model_dir, 'best_model.zip')
            torch.save(model,best_model_path)

        else:
            count +=1

        if count > 5 and epoch > 30:# and epoch > 50:
            
            break

                    

    return best_f1, best_pre, best_rec, best_acc

if __name__ == "__main__":
    
    args = get_args()
    cfg, save_dir, name, epochs, batch_size, k = \
        args.cfg, args.save_dir, args.name, args.epochs, args.batch_size, args.k
    
    params = Params(cfg)
    data_dir, fault_flags, time_steps, channels, ab_range, sli_time, normal_use, threshold_data, ratio = \
        params.data_dir, params.fault_flags, params.time_steps, params.channels, params.ab_range, \
        params.sli_time, params.normal_use, params.threshold_data, params.ratio
    
    #ratio: intersection ratio in the dataset division for sliding windows
    #channels: origianl channels of data
    #threshold_data: value for filtering the channels without sufficient variables
    #k: k-fold cross validation
    #sli_time: sliding end time 
    #normal_use: whether use normal for training


    #introdece the noise for the datasets
    snr = 50
    channel_snr = None
    noise_type = 'Guassian'
    pooling_name = 'max' # 'ava' None
    normal_use = False
    torch.autograd.set_detect_anomaly(True)



    #threshold_data = 0.0##################################
    #fault_flags = [  '12_FLW4_1', '31_GCI'] ########################
    #fault_flags = [ '29_VLVBPS']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if pooling_name == 'ava':
        pooling = nn.AdaptiveAvgPool1d(time_steps//10).to(device)
    elif pooling_name == 'max':
        pooling = nn.AdaptiveMaxPool1d(time_steps//10).to(device)
    else:
        pooling = None

    


    #m_names = ['seonly0', 'seonly1','seonly2','saonly0','saonly1','saonly2','cnn','lstmonlly'] #'attention'
    #m_names = ['lstmonly', 'seonly2','saonly2','cnn'] #'seonly0','saonly2',
    m_names = ['mix-all','mix-paper','mix-gru-se','mix-tcn-se']#,'dlinear']  #'timesnet' 'dlinear'
    m_names = ['timesnet' , 'dlinear', 'conv1d']
    m_names = [ 'conv1d']
    m_names = ['mix-paper']
    m_names = ['mix-all','mix-paper','mix-gru-se','mix-tcn-se','tcn','conv1d','seonly', 'dlinear', 'timesnet']#,'dlinear']  #'timesnet' 'dlinear'
    m_names = ['mix-tcn-se']
    m_names = ['mix-all','mix-paper','mix-gru-se','tcn','conv1d','seonly', 'dlinear', 'timesnet']#,'dlinear']  #'timesnet' 'dlinear'
    m_names = ['mix-gru-se','mix-all','mix-paper','tcn','conv1d','seonly', 'dlinear', 'timesnet']#,'dlinear']  #'timesnet' 'dlinear'
    m_names = ['tcn','conv1d','seonly', 'dlinear', 'timesnet','mix-tcn-se']#
    m_names = ['tcn','mix-tcn-se','mix-all','mix-paper'] # 'dlinear', 'timesnet',
    m_names = ['mix-all','mix-paper','mix-gru-se','mix-tcn-se','tcn','conv1d','seonly'] #, 'dlinear', 'timesnet']
    m_names1 = [ 'dlinear', 'timesnet']
    m_names = ['tcn']
    #m_names = ['mix-gru-se','mix-all','mix-paper']
    #without soft contrastive
    save_dir = './log/codetesting' #'./log/10cls' 
    save_dir = '/mnt/database/torch_results/tanaka-try/uea-re'
    loss_m_list = ['center-zerograd' , 'softmax','mixed']
    loss_m_list = ['sphere-softmax']
    loss_m_list = ['soft-contra','hard-contra']
    loss_m_list = ['center-normalized','softmax']
    loss_m_list = ['center-aug-normalizaed','softmax-aug']
    mask_ratio_set = 0.3 #7 #0.7 all samples used for training  
    loss_m_list = [f'design-ratio{mask_ratio_set}'] #'design'
    loss_m_list = ['center-zerograd','soft-contra', 'design-maskedonlyforcls-.7','softmax','hard-contra','mixed']
    loss_m_list = [ 'softmax-re', 'design-maskedonlyforcls-.7-real',]
    loss_m_list = [ 'softmax-re', 'design-maskedonlyforcls-.7-real',]
    loss_m_list = ['mixed-mask'] #only used known information for training using the same rank dataset
    #loss_m_list = ['center-zerograd','soft-contra'] #, 'mixed']

    datasetss_name = [
            'AtrialFibrillation', #error index in softmax:
            'ArticularyWordRecognition',
            'BasicMotions',
            'CharacterTrajectories',
            'Cricket',
            'DuckDuckGeese',
            'Epilepsy',
            'ERing',
            'EthanolConcentration',
            'FaceDetection',
            'FingerMovements',
            'HandMovementDirection',
            'Handwriting',
            'Heartbeat',
            'InsectWingbeat',
            'JapaneseVowels',
            'Libras',
            'LSST',
            'MotorImagery',
            'NATOPS',
            'PEMS-SF',
            'PenDigits',
            'Phoneme',
            'RacketSports',
            'SelfRegulationSCP1',
            'SelfRegulationSCP2',
            'SpokenArabicDigits',
            'StandWalkJump',
            'UWaveGestureLibrary',
        ]# overed

    datasetss_name1 = [    
            'Phoneme',
            'RacketSports',
            'SelfRegulationSCP1',
            'SelfRegulationSCP2',
            'SpokenArabicDigits',
            'StandWalkJump',
            'UWaveGestureLibrary',]
    #loss_m_list = ['sphere-aug','sphere'] #['softmax-aug','center-aug',]
    #loss_m_list = ['mixed']
    loss_s_m = None #'sphere'
    #loss_m_list = ['mixed','center','softmax','contrastive','supervised-contrastive',] #'center'
    
    noise_intro=False
    normal_use= False #
    if normal_use:
        fault_flags.append('Normal')

    load_filelist = True if os.path.exists(save_dir) else False

    for mask_ratio in [0.3, 0.9, 0.7]:#[50, 100, 200, 500,10]: #50, 100, 200, 10 time steps used in the experiments no use 

      channels = params.channels
      f1s, pres, recs, accs = {}, {}, {}, {}     
      
      #first divide the dataset for k-fold cross validation
      #maxi, mini = torch.tensor(accdataset.maxi.reshape(-1,1)), torch.tensor(accdataset.mini.reshape(-1,1))
      
      for name in datasetss_name: 
        #for each name of the uea datasets, accord to different classification task
        #datasets = accdataset.generate_datasets(fault_flags,val)#if load, return the loaded idx, else generate new idx
        train_data, train_labels, test_data, test_labels = load_UEA(name)
        #mini = np.minimum(train_data.min(axis=(0,1)),test_data.min(axis=(0,1)) )
        #maxi = np.maximum(train_data.max(axis=(0,1)),test_data.max(axis=(0,1)) )
        mini = np.minimum(np.nanmin(train_data, axis=(0, 1)), np.nanmin(test_data, axis=(0, 1)))
        maxi = np.maximum(np.nanmax(train_data, axis=(0, 1)), np.nanmax(test_data, axis=(0, 1)))

        train_data = (train_data-mini)/(maxi-mini)
        test_data = (test_data-mini)/(maxi-mini)
        train_data =train_data.transpose(0,2,1)
        test_data = test_data.transpose(0,2,1)
        assert train_data.ndim == 3 

        
        for loss_m in loss_m_list:
            print(loss_m, 'start training')
            if 'mask' in loss_m:
                #mask_ratio = mask_ratio_set
                result_dir = os.path.join(save_dir, name)
                mask_filelist = os.path.join(result_dir, f'mask_{mask_ratio_set}.txt')
                if not os.path.exists(mask_filelist):
                    if not os.path.exists(os.path.join(result_dir, f'mask_{mask_ratio_set}.txt')) and mask_ratio_set!=0: # and i ==0:
                            
                        num_mask = int(len(train_data)* mask_ratio_set)
                        mask_index = torch.randperm(len(train_data))[:num_mask].tolist()
                        with open(mask_filelist, 'w+') as f:
                            f.write(str(mask_index)+'\n') #save first
                else:
                    with open(mask_filelist,'r') as f:
                        line = f.readlines()[-1]
                        mask_index = line.strip('\n').strip(']').strip('[').replace("'",'').replace(" ",'').split(',')
                        mask_index = [int(_) for _ in mask_index]
            else:
                mask_ratio=0
                mask_index=None

            #mask_ratio=mask_ratio_set
            print(f'loss name {loss_m} in dataset {name} dataset size {train_data.shape}')
            train_loader = load_datasets(train_data=train_data, train_labels=train_labels, device=device, mask_ratio=mask_ratio, mask_index=mask_index)

            test_loader = load_datasets(test_data, test_labels, device=device, mask_ratio=0, mask_index=None)
            #np.save()
            fault_flags=np.arange(train_labels.max()+1)
            print(loss_m)
            for m in m_names:
                print(m)
                result_dir = os.path.join(save_dir, name, loss_m, m)
                print('Expriment of the uea dataset ', result_dir)
                if 'softmax' in loss_m:
                    pooling = None
                f1, pre, rec, acc = train(m=m,  
                    epochs= epochs, #epochs, ##############""""""""""""""
                    batch_size=batch_size, 
                    time_steps=time_steps, 
                    channels=train_data.shape[1], 
                    result_dir=result_dir,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    fault_flags=fault_flags,
                    device=device,
                    loss_m=loss_m,
                    pooling=pooling)
                
                if loss_m not in f1s.keys() :
                    for _ in (f1s, pres, recs, accs):
                        _[loss_m] = {} 

                if m not in f1s[loss_m].keys() :
                    f1s[loss_m][m] = [f1]
                    pres[loss_m][m] = [pre]
                    recs[loss_m][m] = [rec]
                    accs[loss_m][m] = [acc]
                else:
                    f1s[loss_m][m].append(f1)
                    pres[loss_m][m].append(pre)
                    recs[loss_m][m].append(rec)
                    accs[loss_m][m].append(acc)


                final_results_path = os.path.join(result_dir, 'final_avg_acc.txt')
                with open(final_results_path, 'a') as f:
                    f.write('final f1: {} {}\n '.format(sum(f1s[loss_m][m])/len(f1s[loss_m][m]),f1s[loss_m][m]))
                    f.write('final pre: {} {}\n '.format(sum(pres[loss_m][m])/len(pres[loss_m][m]),pres[loss_m][m]))
                    f.write('final rec: {} {}\n '.format(sum(recs[loss_m][m])/len(recs[loss_m][m]),recs[loss_m][m]))
                    f.write('final acc: {} {}\n '.format(sum(accs[loss_m][m])/len(accs[loss_m][m]),accs[loss_m][m]))



