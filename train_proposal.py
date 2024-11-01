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
#from torchsummary import summary
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda import amp
#from datasets_proposal_contrasive import ACCORDDataset, TorchDataset #input dimension: [normal, anomlay, label]
from datasets import ACCORDDataset, TorchDataset
import time
from utils import Params
from networks_torch import CNN1D, TemCNN1D, SaCNN1D, SeCNN1D, LSTMonly, SaCNN1Donly, SeCNN1Donly
from network_contrasive import SeCNN1Dcontrasive, SupConLoss
from networks_torch_newmodel import TemTCN, TemSeTCN
from networks_dl import DLinear
from networks_timesnet import TimesNet
import gc 
from sklearn.metrics import confusion_matrix 
from network_sphere import SeCNN1Dshpere, SeCNN1Dshperewo
from network_contrasive_mixed import hierarchical_contrastive_loss, take_per_row
from network_spatio_temporal import TSEncoder, Classifer
from network_propose_contrasive_center import center_contrasive_loss, SphereClassifier, SeCNN1Donlypropose
from loss_propose import IntraDistanceLoss
#utilize different learning approach for contrasive learning loss based function
#for supervised contrasive learning: no need for softmax function
#to construct the input of the loss function (need to be revised) (randomly select the sub-seires in the input data)
#introduce the center and radius within one sphere and introduce the intra distance to make the data combined more within one sphere
import torch.nn.functional as F

def each_class(labels,result,fault_flags):
    #print('each class: ', fault_flags)
    #print('label: ' , set(labels))
    results = ''
    matrix = confusion_matrix(labels,result)
    #print(matrix)
    batch_sum = matrix.flatten()
    batch_sum = np.sum(batch_sum)
    f1s,pres,recs,accs = [], [], [], []
    for fault_flag in fault_flags:
        index = fault_flags.index(fault_flag)
        #print('index: ',index)
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

def get_args():
    parser = argparse.ArgumentParser(description='ACCORD data training.')
    parser.add_argument('--cfg', type=str, default='./cfg/accord.yml', help='The path to the configuration file.')
    #parser.add_argument('--save-dir', type=str, default='./modeltestingnoise/batchnormwithactivationtransposewithoutdropout', help='The directory of log files.')
    parser.add_argument('--save-dir', type=str, default='/mnt/database/torch_results/proposal/intradistance', help='The directory of log files.')#./modeltestingsnr50/temtcnv3
    #/mnt/database/results_accord_torch/sekiguchi/timesnet1e5 #/contrasive_revised
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied') #noiseintro080
    parser.add_argument('--epochs', type=int, default=500) ################
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--k', type=int, default=10, help='K-fold cross validation.')

    return parser.parse_args()

def train(**kwargs):
    m = kwargs.get('m',None) #model_name
    epochs = kwargs.get('epochs', 1)
    channels = kwargs.get('channels',20)
    batch_size = kwargs.get('batch_size', 8)
    time_steps = kwargs.get('time_steps', 30)
    result_dir = kwargs.get('result_dir', None)
    datasets = kwargs.get('datasets', None)
    train_loader = kwargs.get('train_loader',None)
    test_loader = kwargs.get('test_loader',None)
    fault_flags = kwargs.get('fault_flags',None)
    device = kwargs.get('device','cpu')
    maxi = kwargs.get('maxi', None)
    mini = kwargs.get('mini', None)
    print(device)

    # scheduler
    #scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
  
    if m == 'seonly':
        model = SeCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)
    elif m == 'saonly':
        model = SaCNN1Donly(time_steps=time_steps, channels=channels, classes = len(fault_flags), pos=2)
    elif m == 'cnn':
        model = CNN1D(time_steps=time_steps, channels=channels, classes = len(fault_flags))
    elif m == 'lstmonly':
        model = LSTMonly(time_steps=time_steps, channels=channels, classes = len(fault_flags))
    elif m == 'attention':
        model = TemCNN1D(time_steps=time_steps, channels=channels, classes = len(fault_flags))
    elif m == 'dlinearnoindividual':
        model = DLinear(time_steps=time_steps, channels=channels, classes = len(fault_flags))
    elif m == 'timesnet':
        model = TimesNet(time_steps=time_steps, channels=channels, classes = len(fault_flags))
    elif m == 'temtcn':
        model = TemTCN(time_steps=time_steps,channels=channels, classes = len(fault_flags))
    elif m == 'sesphere':
        model = SeCNN1Dshpere(time_steps=time_steps,channels=channels, classes = len(fault_flags))
    elif m == 'sespherewo':
        model = SeCNN1Dshperewo(time_steps=time_steps,channels=channels, classes = len(fault_flags))
    elif m == 'secontrasivewopool':
        model = SeCNN1Dcontrasive(time_steps=time_steps,channels=channels, classes = len(fault_flags))
        classifer = SupConLoss(contrast_mode='all').to(device) 
    elif 'temsetcn' in m:
        pos = int(m[-1])
        model = TemSeTCN(time_steps=time_steps,channels=channels, classes = len(fault_flags),pos = pos)
    elif 'mix' in m:
        model = TSEncoder(time_steps=time_steps,channels=channels, classes = len(fault_flags)) #with or without the classifer
    elif 'propose' in m:
        model = SeCNN1Donlypropose(time_steps=time_steps,channels=channels, classes = len(fault_flags))
        classifer = SphereClassifier(num_classes=len(fault_flags), feature_dim=128).to(device)
        classifer = IntraDistanceLoss(num_classes=len(fault_flags), feature_dim=128).to(device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer_model = optim.Adam(model.parameters(), lr=0.01)
    optimizer_center = optim.Adam(classifer.parameters(), lr=0.001)
    count = 0
    best_f1 = 0
    best_epoch = 0


    for epoch in range(epochs):
        y_pred = None
        y_true = None
        epoch_loss = 0
        epoch_accuracy = 0
        labels_train = None
        data_train = None
        labels = None
        
        train_start_time = time.time()
        for data, label in tqdm(train_loader):
            #labels_train = None
            #data_train = None
        
            #data = ((data- mini.repeat(2,1)) / (maxi.repeat(2,1)-mini.repeat(2,1))).float().to(device) #2B,C,T anomaly + normal
            data = ((data- mini) / (maxi-mini)).float().to(device)
            #label = label.mean(dim=-1,keepdim=True)
            label = label.type(torch.LongTensor).to(device)#.float() #need to transfer it into the one window one label .min(axis=-1).values
            labels = label if labels is None else torch.cat((labels,label))
            
 
            optimizer_model.zero_grad()
            optimizer_center.zero_grad()

            features, cls = model(data)
            #logits, distances = classifer(features,label)
            #diff, sphere_loss, center_loss = classifer(features,label)
            #loss = classifer.compute_loss(logits, distances, label)
            if epoch < 5:
                loss = criterion(cls,label)
            else:
                loss = classifer(features,label) + criterion(cls,label) #.compute_loss(cls, label, sphere_loss, center_loss)
            #criterion(cls,labels) 
            assert torch.isfinite(loss).item()
            loss.backward()

            optimizer_model.step()
            optimizer_center.step()
            #if two stage: trained backbone used for classifer 
            if labels_train is None:
                labels_train = label.flatten()#label_f1.repeat(3,1).flatten()
                data_train =torch.argmax(cls,1).flatten()
                #former version output
            else:
                labels_train = torch.cat([labels_train,label.flatten()])#label_f1.repeat(3,1).flatten()])
                data_train = torch.cat([data_train,torch.argmax(cls,1).flatten()])
            acc = (cls.argmax(dim=1) == label).float().mean()
        #acc = (data_train == labels_train).float().mean()
        #acc = (output.argmax(dim=1) == sup_labels).float().mean()
        #acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc.detach().item() / len(train_loader)
            epoch_loss += loss.detach().item() / len(train_loader)
        print(torch.unique(labels))
        print(torch.unique(labels_train))
        train_end_time = time.time()
        training_time = train_end_time - train_start_time
        labels_train = labels_train.cpu()
        data_train = data_train.cpu()
        _, matrix, train_F1, train_pre, train_rec = each_class(labels_train,data_train,fault_flags)       

        for name, params in model.named_parameters():
            #if params.grad is None:
            print(f'grad of {name} is {params.grad} ')
        
        for name, params in classifer.named_parameters():
            if params.grad is not None:
                print(f'grad of {name} is {params.grad} ')
        
        each_cls_path = os.path.join(result_dir, 'train_each.txt')
        with open(each_cls_path, 'a+') as f:
            f.write('Epoch:{}\n'.format(epoch+1))
            f.write(str(_))
            f.write(str(matrix)+'\n')
        
        with torch.no_grad():
            
            test_start_time = time.time()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in tqdm(test_loader):
                #data = ((data- mini.repeat(2,1)) / (maxi.repeat(2,1)-mini.repeat(2,1))).to(device).float()
                data = ((data- mini) / (maxi-mini)).float().to(device)
                #label = label.mean(dim=-1,keepdim=True)
                label = label.type(torch.LongTensor).to(device) #.min(axis=-1).values
                #label = label.type(torch.LongTensor).to(device)

                #features = model(data)
                features,cls = model(data)
                #logits, distances = classifer(features)
                loss = F.cross_entropy(cls, label)#classifer.compute_loss(logits, distances,label)
                assert torch.isfinite(loss).item()
        

                outputs = torch.argmax(cls,1)#not trained the final layer of the fc layer
                if y_pred is None:
                    y_pred = outputs.flatten()
                    y_true = label.flatten()
                else:
                    y_pred = torch.cat([y_pred,outputs.flatten()])
                    y_true = torch.cat([y_true,label.flatten()])
                
                acc = (cls.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc.detach().item() / len(test_loader)
                epoch_val_loss += loss.detach().item() / len(test_loader)
                
            test_end_time = time.time()
            testing_time = test_end_time - test_start_time

            gc.collect()

        y_pred = y_pred.cpu()
        y_true = y_true.cpu()
        _, matrix, avg_F1, avg_pre, avg_rec = each_class(y_true,y_pred,fault_flags)
        
        postfix = 'Epoch: {0}, Val F1: {1:.4f}, Val acc: {2:.4f}'.format(epoch+1, avg_F1, epoch_val_accuracy)
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
            #f.write('Center: {}\n'.format(epoch+1))

    
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

        if count > 5 and epoch > 50:
            
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
    #fault_flags = [  '12_FLW4_1', '31_GCI'] ##############
    #fault_flags = [ '29_VLVBPS']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    save_dir = '/mnt/database/torch_results/proposal/distancecontrasiveloss1000anomalysamples' 
    save_dir = '/mnt/database/torch_results/proposal/intradistance-centerbased'
    #save_dir = 'mnt/database/torch_results/proposal/codetesing/intradis'
    #m_names = ['seonly0', 'seonly1','seonly2','saonly0','saonly1','saonly2','cnn','lstmonlly'] #'attention'
    #m_names = ['lstmonly', 'seonly2','saonly2','cnn'] #'seonly0','saonly2',
    m_names = ['secontrasivewopool'] #['dlinearnoindividual']# ['dlinear'] #['attention','seonly','saonly','timesnet','dlinear']#['attention','seonly','saonly']
    #m_names = ['attention'] #
    #m_names = ['temtcn']
    #m_names = ['cnn']
    #m_names = ['temsetcn3','temsetcn2','temsetcn2']
    #m_names = ['temsetcn5','temsetcn4']
    m_names1  = ['sespherewo']#['sesphere']
    m_names  = ['cnn']
    m_names = ['mixnet-10layer-onestage']
    m_names = ['se-propose-cosdis']
    m_names= ['se-propose-msereplace1d']
    if normal_use:
        fault_flags.append('Normal')
    '''
    if not os.path.exists(os.path.join(save_dir,time_steps,'data')):
        load_filelist = False
    else:
        load_filelist = True
    '''
    load_filelist = True if os.path.exists(save_dir)  else False
    #load_filelist = False
    #True
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
                                sli_time=2000,#sli_time, 
                                save_dir=save_dir,
                                snr=snr,
                                channel_snr=channel_snr,
                                noise_type=noise_type,
                                load_filelist=load_filelist,
                                noise_intro=True, 
                                normal_increase=True)
      
      #first divide the dataset for k-fold cross validation
      #maxi, mini = torch.tensor(accdataset.maxi.reshape(-1,1)), torch.tensor(accdataset.mini.reshape(-1,1))

      for val in range(k): 
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
        train_loader = DataLoader(dataset=TorchDataset(file_list=datasets['train'],file_dir=save_dir, time_steps=time_steps), batch_size=8,drop_last=True,pin_memory=True, num_workers=4)
        test_loader = DataLoader(dataset=TorchDataset(file_list=datasets['test'],file_dir=save_dir, time_steps=time_steps), batch_size=8,drop_last=True,pin_memory=True, num_workers=4)

        print(save_dir)
        for m in m_names:
            result_dir = os.path.join(save_dir,str(time_steps), str(val), m)
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
                mini=mini)
            
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


