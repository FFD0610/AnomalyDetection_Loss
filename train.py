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
from datasets import ACCORDDataset, TorchDataset
import time
from utils import Params, each_class
import gc 
from losses.center_loss import CenterLoss
from losses.s_contrastive import  SupConLoss
from losses.mixed_loss import Mixed_calculation
from losses.asoftmax import SphereFace
from losses.soft_contrastive import SoftContra, HardContra
from losses.auge import Augement
from networks import *
'''
updated:
1. intro models to verify the loss function ( dlinear + timesnet)
2. softmax loss revision (check why )
3. se-based model cannot work for constrative loss based trainin process -> ensure why
4. center loss not best - revise this one 

'''
#updated:
'''
1. two new contrastive loss function 
2. center loss with normalized center (self)
3. pooling layer introduced into the model verification
'''

#updated 11.14
'''
1. use pooling layer for softmax loss and center loss 
2. More features for learning representation 

'''
#updated 11.20
'''
1. data augementation in softmax loss and center loss 
2. softmax replaced by asoftmax loss (let it sepearable by introducing the margin)
3. try other output of the pooling layer
'''
#updated in 0203
'''
deal with the situation some samples with label but other without 
the dataset with label should be minimum to the ratio to the number without labels
so the dataset set with label and without label should be settled before input into the model training process 

'''
def get_args():
    parser = argparse.ArgumentParser(description='ACCORD data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/accord.yml', help='The path to the configuration file.')
    parser.add_argument('--save-dir', type=str, default='/mnt/database/torch_results/codetesting' , help='The directory of log files.')#./modeltestingsnr50/temtcnv3
    #/mnt/database/results_accord_torch/sekiguchi/timesnet1e5
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied') #noiseintro080
    parser.add_argument('--epochs', type=int, default=200) ################
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
    noise_intro = kwargs.get('noise_intro',True)
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
    


    if 'aug' in loss_m:
        loss_aug = Augement(pooling=pooling)
    

         
    model = model.to(device)
    loss_s = loss_s.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    count = 0
    best_f1 = 0

    for epoch in range(epochs):
        y_pred,y_true = None, None
        epoch_loss,epoch_accuracy = 0, 0
        labels_train,data_train = None, None

        
        train_start_time = time.time()
        for data, label in tqdm(train_loader):
            #data is composed of the noise-intro and original part
            if noise_intro: 
                data = data[:,:channels]
                #('data size: ',data.size())
            data = ((data- mini) / (maxi-mini)).float().to(device)
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
                else:
                    loss_c_score, diff, cls, label = loss_c(data=data, labels=label, model=model)

            #features, cls = features.to(device), cls.to(device)
                #print(label.size(),cls.size())
                loss_s_score = loss_s(cls,label) if 'aug' not in loss_m  else loss_s(cls,label) + loss_s(cls_aug,label_aug)
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
                    
                label = label.type(torch.LongTensor).to(device)
                if 'sphere' in loss_m:
                    loss = loss_s(loss_cls(embed,label),label) if 'aug' not in loss_m  else loss_s(embed,label) + loss_s(embed_aug,label_aug)
                else:
                    loss = loss_s(cls,label) if 'aug' not in loss_m  else loss_s(cls,label) + loss_s(cls_aug,label_aug)
                


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
        _, matrix, train_F1, train_pre, train_rec = each_class(labels_train,data_train,fault_flags)

        each_cls_path = os.path.join(result_dir, 'train_each.txt')
        with open(each_cls_path, 'a+') as f:
            f.write('Epoch:{}\n'.format(epoch+1))
            f.write(str(_))
            f.write(str(matrix)+'\n')

        #evaluation 
        with torch.no_grad():
            
            test_start_time = time.time()
            epoch_val_accuracy = 0
            epoch_val_loss = 0

            for data, label in tqdm(test_loader):
                if noise_intro:
                    data = data[:,channels:]                    
                data = ((data- mini) / (maxi-mini)).to(device).float()
                
                features, cls = model(data)
                features, cls = features.to(device), cls.to(device)

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
            f"Model : {m} Time_Steps: {time_steps} Val_idx : {val} Epoch : {epoch+1} - loss : {epoch_loss:.4f} \
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

        if count > 5:# and epoch > 50:
            
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


    
        

    #threshold_data = 0.0##################################
    #fault_flags = [  '12_FLW4_1', '31_GCI'] ##############
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
    m_names = ['mix-all','mix-paper','mix-gru-se','mix-tcn-se','tcn','conv1d','seonly', 'dlinear', 'timesnet']
    #m_names = ['mix-gru-se','mix-all','mix-paper']
    save_dir = '/mnt/database/torch_results/tanaka-try/10cls' #tanaka-try' 

    loss_m_list = ['center-zerograd' , 'softmax','mixed']
    loss_m_list = ['sphere-softmax']
    loss_m_list = ['soft-contra','hard-contra']
    loss_m_list = ['center-normalized','softmax']
    loss_m_list = ['center-aug-normalizaed','softmax-aug']
    #loss_m_list = ['sphere-aug','sphere'] #['softmax-aug','center-aug',]
    #loss_m_list = ['mixed']
    loss_s_m = None #'sphere'
    #loss_m_list = ['mixed','center','softmax','contrastive','supervised-contrastive',] #'center'

    noise_intro=True
    normal_use= True #
    if normal_use:
        fault_flags.append('Normal')

    load_filelist = True if os.path.exists(save_dir) else False
    for time_steps in [50]:#[50, 100, 200, 500,10]: #50, 100, 200, 10 time steps used in the experiments

      channels = params.channels
      f1s, pres, recs, accs = {}, {}, {}, {}     
      accdataset = ACCORDDataset(data_dir=data_dir, 
                                fault_flags=fault_flags,
                                ab_range=ab_range,
                                time_steps=time_steps,
                                channels=channels,
                                k=k,
                                threshold_data = threshold_data,
                                normal_use=normal_use, 
                                ratio=ratio, 
                                sli_time=sli_time, 
                                save_dir=save_dir,
                                snr=snr,
                                channel_snr=channel_snr,
                                noise_type=noise_type,
                                load_filelist=load_filelist,
                                noise_intro=noise_intro,#False, 
                                normal_increase=True,
                                ) #False)
      
      #first divide the dataset for k-fold cross validation
      #maxi, mini = torch.tensor(accdataset.maxi.reshape(-1,1)), torch.tensor(accdataset.mini.reshape(-1,1))

      for val in range(k): 
        datasets = accdataset.generate_datasets(fault_flags,val)
        for loss_m in loss_m_list:
            for m in m_names:
                result_dir = os.path.join(save_dir,str(time_steps), loss_m, str(val), m)
                print(result_dir)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)

                val_filelist = os.path.join(result_dir, 'val_list.txt')
                with open(val_filelist, 'a') as f:
                    f.write('Val:{}\n'.format(val))
                    f.write(str(datasets['test'])+'\n') #save first

      for val in range(k): #k fold verification experiments
        datasets = accdataset.generate_datasets(fault_flags,val)
        maxi, mini = torch.tensor(datasets['maximini'][0].reshape(-1,1)), torch.tensor(datasets['maximini'][1].reshape(-1,1))
        if not load_filelist : 
            print(accdataset.r_ranges)
            channels = accdataset.get_channels() 
            
        else: 
            print(maxi)
            channels = maxi.shape[0]
        print(channels)
        train_loader = DataLoader(dataset=TorchDataset(file_list=datasets['train'],file_dir=save_dir, time_steps=time_steps), batch_size=8,drop_last=True,pin_memory=True, num_workers=8)
        test_loader = DataLoader(dataset=TorchDataset(file_list=datasets['test'],file_dir=save_dir, time_steps=time_steps), batch_size=8,drop_last=True,pin_memory=True, num_workers=8)

        print(save_dir)
        for loss_m in loss_m_list:
            print(loss_m)
            for m in m_names:
                print(m)
            
                result_dir = os.path.join(save_dir,str(time_steps), loss_m, str(val), m)
                f1, pre, rec, acc = train(m=m,  
                    epochs=epochs, 
                    batch_size=batch_size, 
                    time_steps=time_steps, 
                    channels=channels, 
                    result_dir=result_dir,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    fault_flags=fault_flags,
                    device=device,
                    maxi=maxi,
                    mini=mini,
                    noise_intro=noise_intro,
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

                if val == k-1:  
                    final_results_path = os.path.join(result_dir, 'final_avg_acc.txt')
                    with open(final_results_path, 'a') as f:
                        f.write('final f1: {} {}\n '.format(sum(f1s[loss_m][m])/len(f1s[loss_m][m]),f1s[loss_m][m]))
                        f.write('final pre: {} {}\n '.format(sum(pres[loss_m][m])/len(pres[loss_m][m]),pres[loss_m][m]))
                        f.write('final rec: {} {}\n '.format(sum(recs[loss_m][m])/len(recs[loss_m][m]),recs[loss_m][m]))
                        f.write('final acc: {} {}\n '.format(sum(accs[loss_m][m])/len(accs[loss_m][m]),accs[loss_m][m]))


