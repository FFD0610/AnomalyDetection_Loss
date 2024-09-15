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
from torch.cuda import amp
from datasets import ACCORDDataset, TorchDataset
import time
from utils import Params, each_class
import gc 
from losses.center_loss import CenterLoss
from losses.s_contrastive import  SupConLoss
from losses.mixed_loss import Mixed_calculation
from networks import *

def get_args():
    parser = argparse.ArgumentParser(description='ACCORD data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/accord.yml', help='The path to the configuration file.')
    parser.add_argument('--save-dir', type=str, default='/mnt/database/torch_results/codetesting' , help='The directory of log files.')#./modeltestingsnr50/temtcnv3
    #/mnt/database/results_accord_torch/sekiguchi/timesnet1e5
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied') #noiseintro080
    parser.add_argument('--epochs', type=int, default=2) ################
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
    loss_m = kwargs.get('loss_m',None)
    print(device)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
  
    if m == 'seonly':
        model = SeCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)
        feat_dim = 128
    
    elif m == 'saonly':
        model = SaCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)

    elif m == 'dlinear':
        model = DLinear(time_steps=time_steps, channels=channels, classes = len(fault_flags))
        feat_dim = 20

    elif m == 'timesnet':
        model = TimesNet(time_steps=time_steps, channels=channels, classes = len(fault_flags))
        feat_dim = 32

    elif 'mix' in m:
        if 'all' in m :
            model = MixedNet(time_steps=time_steps,channels=channels, classes = len(fault_flags))
            feat_dim = 128+32+128
        if 'gru-se' in m:   
            model = MixedNet2(time_steps=time_steps,channels=channels, classes = len(fault_flags))
            feat_dim = 128+128
        if 'tcn-se' in m:
            model = MixedNet3(time_steps=time_steps,channels=channels, classes = len(fault_flags))
            feat_dim = 128+32
        else: 
            model = TSEncoder(time_steps=time_steps,channels=channels, classes = len(fault_flags))
            feat_dim = 128+32

    model = model.to(device)
    #feat_dim = 128 if 'mix' not in m else feat_dim

    #loss definition
    loss_s = nn.CrossEntropyLoss().to(device)
    if loss_m == 'center':
        loss_c = CenterLoss(feat_dim=feat_dim, classes=len(fault_flags))
        optimizer_center = optim.Adam(loss_c.parameters(), lr=0.001)
    elif loss_m == 'contrastive':
        loss_c = SupConLoss(contrast_mode='one')

    elif loss_m == 'supervised-contrastive':
        loss_c = SupConLoss(contrast_mode='all')

    elif loss_m == 'mixed':
        loss_c = Mixed_calculation()
    
         
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

            #all of the model have the two output: embedding features and the classification results
            if 'softmax' not in loss_m:
                loss_c_score, diff, cls, label = loss_c(data=data, labels=label, model=model)

            #features, cls = features.to(device), cls.to(device)
                loss = loss_c_score + loss_s(cls,label)
            else:
                _, cls = model(data)
                label = label.type(torch.LongTensor).to(device)
                loss = loss_s(cls,label)

            loss.backward()
            optimizer.step()
            if loss_m == 'center':
                
                optimizer_center.step()
                loss_c.update_centers(diff)


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
                label = label.type(torch.LongTensor).to(device)
                features, cls = model(data)
                features, cls = features.to(device), cls.to(device)

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

    #threshold_data = 0.0##################################
    fault_flags = [  '12_FLW4_1', '31_GCI'] ##############
    #fault_flags = [ '29_VLVBPS']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    #m_names = ['seonly0', 'seonly1','seonly2','saonly0','saonly1','saonly2','cnn','lstmonlly'] #'attention'
    #m_names = ['lstmonly', 'seonly2','saonly2','cnn'] #'seonly0','saonly2',
    m_names = ['mix-all','mix-paper','mix-gru-se','mix-tcn-se']#,'dlinear']  #'timesnet' 'dlinear'
    save_dir = '/mnt/database/torch_results/codetesting' 
    loss_m_list = ['mixed','center','softmax','contrastive','supervised-contrastive',] #'center'
    noise_intro=True

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
                                normal_increase=True) #False)
      
      #first divide the dataset for k-fold cross validation
      #maxi, mini = torch.tensor(accdataset.maxi.reshape(-1,1)), torch.tensor(accdataset.mini.reshape(-1,1))

      for val in range(2): 
        datasets = accdataset.generate_datasets(fault_flags,val)
        for m in m_names:
          result_dir = os.path.join(save_dir,str(time_steps), str(val), m)
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
                    loss_m=loss_m)
                
                if m not in f1s.keys():
                    f1s[m] = [f1]
                    pres[m] = [pre]
                    recs[m] = [rec]
                    accs[m] = [acc]
                else:
                    f1s[m].append(f1)
                    pres[m].append(pre)
                    recs[m].append(rec)
                    accs[m].append(acc)

                if val == k-1:  
                    final_results_path = os.path.join(result_dir, 'final_avg_acc.txt')
                    with open(final_results_path, 'a') as f:
                        f.write('final f1: {} {}\n '.format(sum(f1s[m])/len(f1s[m]),f1s[m]))

                        f.write('final pre: {} {}\n '.format(sum(pres[m])/len(pres[m]),pres[m]))
                        f.write('final rec: {} {}\n '.format(sum(recs[m])/len(recs[m]),recs[m]))
                        f.write('final acc: {} {}\n '.format(sum(accs[m])/len(accs[m]),accs[m]))


