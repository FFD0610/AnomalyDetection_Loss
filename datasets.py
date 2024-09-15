#first divide as r_ranges
#when the features changes evidently, using sliding windsow in the interactivation ratio of 70% for dividing the dataset
import os
import math
import numpy as np
from tqdm import tqdm
import pandas as pd 
import random
from torch.utils.data import Dataset, DataLoader
import torch
import gc

class ACCORDDataset(object):
    def __init__(self, data_dir, fault_flags, ab_range, time_steps, 
                channels, k, threshold_data, normal_use, ratio, 
                sli_time, save_dir, snr, channel_snr, noise_type,
                load_filelist=False, noise_intro= False, normal_increase=False):
        #noise_intro: whether introduce the noise into the dataset or not
        #normal_increase: whether useing the dataset within each abnoraml file from 0 to 1000s (besides the 100 r_range )
        self.data_dir = data_dir # データフォルダ
        self.fault_flags = fault_flags # 故障種別データフラッグ
        self.ab_range = ab_range #abnormal introduced range
        self.normal_use = normal_use #use normal data for test
        self.time_steps = time_steps # データ長
        self.channels = channels
        self.k = k # k-分割交差検証のサブセット数
        self.threshold_data = threshold_data
        # 正常/異常データの読み取り
        self.ratio=ratio #intersection ratio for each sliding window
        self.save_dir = os.path.join(save_dir,str(self.time_steps)) 
        self.sli_time = sli_time #end time for sliding window 
        self.snr = snr
        self.channels_snr = channel_snr
        self.noise_type = noise_type

        self.noise_intro = noise_intro
        if self.noise_intro:
            print('intro noise')
        self.normal_increase = normal_increase

        self.load_filelist = load_filelist
        if self.load_filelist:
            #self.test_sample_list = self.read_val_list(val_idx)
            print('Loading the validatation list from the saved file')
        
        else:
            self.index = self.select(self.data_dir,self.fault_flags) #filter the channels without sufficient variables
            self.samples_list = self._get_samples_list() # for differnt time step, generate the file updately
            
            #if not self.load_filelist:
                #self.r_ranges = self.ge_r_ranges() # return the r_file name for each training and testing 
            self.r_ranges = self.ge_r_ranges() #for each time steps, the validation list is fixed
            assert self.mini is not None



    def _split_files(self, flags): #with normalization
        '''divide the original file and save into different count with sliding windows
           ab_range: [1060,1000]
           from original files load data
           normal data comes from two parts: 1. r_range = 100; 2. other r_range except for the 100, the 0-1000s

           noise will only used for training, not for testing.
           therefore, the saved file will include one with noise and without noise
           but the maxium and minium value of the channel will be utilized
        '''
        npy_list = {}
        count = 0
        save_dir = os.path.join(self.save_dir,'data')
        self.maxi, self.mini = None, None
        noise_arr = None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        flags.sort()
        for id,flag in enumerate(flags):
            if flag == 'Normal':
                continue
            else:
                npy_list[flag] = {}
                files_dir = os.path.join(self.data_dir, flag, str(self.ab_range[0] - self.ab_range[1]))
                r_ranges = os.listdir(files_dir)
                r_ranges.sort()
                r_ranges.remove('100')

                ab_labels = []

                for r_range in tqdm(r_ranges, desc='Sliding the file of flag ({})'.format(flag)):
                    npy_list[flag][r_range] = []
                    file_dir = os.path.join(files_dir,r_range,'OUTPUT.ft92')
                    file_arr = pd.read_csv(file_dir,header = 0)
                    #every file split the useless column
                    file_arr = file_arr.drop(file_arr.columns[self.index],axis=1)
                    file_arr  = file_arr.to_numpy()[:,1:]
                    
                    if self.noise_intro:
                        noise_arr = self.intro_noise(file_arr,self.snr,self.channels_snr,noise_type='Guassian') 

                        self.mini = np.minimum(noise_arr.min(axis=0),self.mini) if self.mini is not None else np.min(noise_arr,axis=0)
                        self.maxi = np.maximum(noise_arr.max(axis=0),self.maxi) if self.maxi is not None else np.max(noise_arr,axis=0)
                    else:
                        self.mini = np.minimum(file_arr.min(axis=0),self.mini) if self.mini is not None else np.min(file_arr,axis=0)
                        self.maxi = np.maximum(file_arr.max(axis=0),self.maxi) if self.maxi is not None else np.max(file_arr,axis=0)

                    ab_labels = id*np.ones( (len(file_arr) - self.ab_range[1], 1) )
                    n_labels = ( len(self.fault_flags)-1) *np.ones((self.ab_range[1], 1) )
                    labels = np.vstack((n_labels,ab_labels))
                    file_arr = np.hstack((file_arr, labels)) #read the file and annotate the file with the corret labels

                    #for anomaly
                    #sliding from the self.range[1] to the self.sliding time
                    for start_id in range(self.ab_range[1], self.sli_time, int((1-self.ratio)*self.time_steps)):
                        if start_id + self.time_steps > self.sli_time:
                            start_id = self.sli_time - self.time_steps
                        
                        sample = np.hstack((noise_arr[start_id:start_id+self.time_steps],file_arr[start_id:start_id+self.time_steps])) \
                                if self.noise_intro else file_arr[start_id:start_id+self.time_steps]
                        #sample_dir = os.path.join(range_dir,'{}.npy'.format(str(count)))
                        sample_dir = os.path.join(save_dir,'{}.npy'.format(str(count)))
                        sample_name = '{}.npy'.format(str(count))
                        npy_list[flag][r_range].append(sample_name)
                        np.save(sample_dir,np.transpose(sample))
                        count += 1

                    random.shuffle(npy_list[flag][r_range]) #####for the only one range results
                    #whether use the [0, 1000]s normal data
                    if self.normal_use and self.normal_increase:
                        for start_id in range(0,self.ab_range[1], int((1-self.ratio)*self.time_steps)):
                            if start_id + self.time_steps > self.ab_range[1]:
                                start_id = self.ab_range[1] - self.time_steps
                            sample = np.hstack((noise_arr[start_id:start_id+self.time_steps],file_arr[start_id:start_id+self.time_steps])) \
                                        if self.noise_intro else file_arr[start_id:start_id+self.time_steps]

                            sample_dir = os.path.join(save_dir,'{}.npy'.format(str(count)))
                            sample_name = '{}.npy'.format(str(count))
                            if 'Normal' not in npy_list.keys():
                                npy_list['Normal'] = []
                            npy_list['Normal'].append(sample_name)
                            np.save(sample_dir,np.transpose(sample))
                            count += 1

                if self.normal_use:
                    #npy_list[flag]['100'] = []
                    starting = 0
                    #100 normal range also saving as 
                    normal_file_dir = os.path.join(self.data_dir, flag, str(self.ab_range[0] - self.ab_range[1]),str(100),'OUTPUT.ft92')
                    normal_file_arr = pd.read_csv(normal_file_dir,header = 0)
                    normal_file_arr = normal_file_arr.drop(normal_file_arr.columns[self.index],axis=1)
                    normal_file_arr = normal_file_arr.to_numpy()[:,1:]

                    if self.noise_intro:
                        noise_arr = self.intro_noise(normal_file_arr,self.snr,self.channels_snr,noise_type='Guassian') 
                        self.mini = np.minimum(noise_arr.min(axis=0),self.mini) if self.mini is not None else np.min(noise_arr,axis=0)
                        self.maxi = np.maximum(noise_arr.max(axis=0),self.maxi) if self.maxi is not None else np.max(noise_arr,axis=0)
                    else:
                        self.mini = np.minimum(normal_file_arr.min(axis=0),self.mini) if self.mini is not None else np.min(normal_file_arr,axis=0)
                        self.maxi = np.maximum(normal_file_arr.max(axis=0),self.maxi) if self.maxi is not None else np.max(normal_file_arr,axis=0)

                    #normal_file_arr = (normal_file_arr[:,1:] - self.mini) / (self.maxi-self.mini)
                    n_labels = (len(self.fault_flags)-1)*np.ones((len(normal_file_arr),1))
                    n_arr = np.hstack((normal_file_arr,n_labels))

                    for start_id in range(0,len(n_arr),self.time_steps):
                        if start_id + self.time_steps > len(n_arr):
                            start_id = len(n_arr) - self.time_steps
                        sample = np.hstack((noise_arr[start_id:start_id+self.time_steps],n_arr[start_id:start_id+self.time_steps])) \
                                    if self.noise_intro else n_arr[start_id:start_id+self.time_steps]
                        sample_dir = os.path.join(save_dir,'{}.npy'.format(str(count)))
                        np.save(sample_dir,np.transpose(sample)) 
                        sample_name = '{}.npy'.format(str(count))
                        if 'Normal' not in npy_list.keys():
                                npy_list['Normal'] = []
                        npy_list['Normal'].append(sample_name)
                        count += 1

        #print('npy list :',npy_list) 
        #npy_list: [flag] : [r_range]-- divided filenames
        gc.collect()
        
        maxi_dir = os.path.join(self.save_dir,'maxi.npy')
        np.save(maxi_dir,np.transpose(self.maxi))
        print('maix_shape: ',self.maxi.shape)
        mini_dir = os.path.join(self.save_dir,'mini.npy')
        np.save(mini_dir,np.transpose(self.mini))
        
        return npy_list

    def _get_samples_list(self):
        return self._split_files(self.fault_flags)

    def save_file_list(self, names,k):
        """save the file first
        """
        for i in range(k):
            val_list,_ = self.ge_val_list(i)
            for name in names:
                save_dir = os.path.join(self.save_dir, str(i), name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                val_filelist = os.path.join(save_dir, 'val_list.txt')
                with open(val_filelist, 'a+') as f:
                    f.write('Val:{}\n'.format(i))
                    f.write(str(val_list)+'\n')

        return val_filelist

    def get_channels(self):
        if self.load_filelist: 
            #if using the saved files 
            channels = self.channels_load
        else:
            channels = self.channels - len(self.index) 
        return channels

    def ge_val_list(self, val_idx): #return divided npy file name of val list
        val_lists = {}
        train_lists = {}
        r_ranges = self.r_ranges
        self.paras = 'val_idx:{} dataset division\n'.format(val_idx)
        for id,flag in enumerate(self.fault_flags):
            val_lists[flag] = None
            if flag == 'Normal':
                continue
            else:
                #read the array of files as the index of it to form the k-fold validation dataset 
                #how to read as the val_list generate evenly into different files and read ?? --> find out the code which can devide the dataset through the dataset 
                val_lists[flag] = r_ranges[flag][:,val_idx]
                train_lists[flag] = [i for i in r_ranges[flag].flatten() if i not in val_lists[flag]]
                para = 'flag{} \ntrain range:{}\ntest range:{}\n'.format(flag,train_lists[flag],val_lists[flag])
                self.paras += para

        return val_lists, train_lists   

    def ge_r_ranges(self): #return the r_file name for each training and testing 
        r_ranges = {}
        for flag in self.fault_flags:
            if flag == 'Normal':
                continue
            else:
                
                r_range = os.listdir(os.path.join(self.data_dir, flag, str(self.ab_range[0] - self.ab_range[1])))
                r_range.remove('100')
                r_range.sort()

                file_num = len(r_range)//self.k
                #read the array of files as the index of it to form the k-fold validation dataset 
                #how to read as the val_list generate evenly into different files and read ?? --> find out the code which can devide the dataset through the dataset 
                r_range = np.array(r_range).reshape(file_num,self.k)
                for i in range(file_num):
                    np.random.shuffle(r_range[i])
                r_ranges[flag] = r_range
        return r_ranges


    def select(self,filedir,filenames):
        '''
            return the index for delete for each type faults
            filedir: directory to the data
        '''
        print('Load the datasets to filter the channles without sufficent variables')

        index = []
        if self.threshold_data == 0.0:
            index = []
            print('all channels preserve')
        else:
            types = os.listdir(filedir) #use all file for selecting
            #types = filenames #only use faults type for selecting 
            for type in types:
                path = os.path.join(filedir,type,str(60))
                filelist = os.listdir(path)
                filelist.remove('100')

                for file in filelist:
                    dir = os.path.join(path,file,'OUTPUT.ft92')
                    f = pd.read_csv(dir,header = 0, skiprows=range(1,1001))
                    for idx,value in enumerate(f.std()):
                        if value < float(self.threshold_data):
                            index.append(idx)

        index = set(index)
        index = list(index)
        index.sort()
        print(index,len(index))
        return index

    def intro_noise(self, x, snr=10, channel_snr = None, noise_type = 'Guassian'):
        # x: singal shape [T,C]
        # snr: singal ratio to noise (power ratio)
        # channel_snr: different snr value for different channels
        # noise_type: different kinds of noisy plan to introduce
        # Ps: power for signal; Pn: power for the noise; 
        Ps = np.sum(abs(x)**2,0)/x.shape[0]
        if channel_snr is None:
            # all the same snr
            Pn = Ps/(10**((snr/10)))
        else: 
            Pn = Ps/(10**((channel_snr/10)))####save this value for final validation

        #generate the noise
        if noise_type == 'Guassian':
            noise = np.tile(np.random.randn(x.shape[0]).reshape(-1,1), ((1,x.shape[1]))) * np.sqrt(Pn) 

        return x + noise

    def generate_datasets(self, flags, val_idx):
        #generate the numpy filenames for test and training  
        self.paras = 'val_idx:{} dataset division\n'.format(val_idx)
        if self.load_filelist:
            self.maxi = np.load(os.path.join(self.save_dir,'maxi.npy'))
            self.mini = np.load(os.path.join(self.save_dir,'mini.npy'))
            print('reading the testing lists...')
            test_sample_list = self.read_val_list(val_idx)
            all_list = os.listdir(os.path.join(self.save_dir,'data'))
            print('reading the maximum and minimum of the dataset')
            '''
            for file in tqdm(all_list, desc='reading the maximum and minimum of the dataset'): 
                file_arr_dir = os.path.join(self.save_dir,'data',file)#'{}'.format(str
                file_arr = np.transpose(np.load(file_arr_dir))
                self.mini = np.minimum(file_arr.min(axis=0),self.mini) if self.mini is not None else np.min(file_arr,axis=0)
                self.maxi = np.maximum(file_arr.max(axis=0),self.maxi) if self.maxi is not None else np.max(file_arr,axis=0)
            '''
            #print(len(test_sample_list),len(all_list))
            self.channels_load = self.maxi.shape[0] -1  #delete the label
            train_sample_list = list(set(all_list)-set(test_sample_list))
            #for item in test_sample_list:
            #    all_list.remove(item)
            #print(len(all_list))
            #train_sample_list = [i for i in test_sample_list if i not in test_sample_list]
            #train_sample_list = [i for i in all_list if i not in test_sample_list]

            return {'train': train_sample_list,
                'test': test_sample_list,
                'maximini':[self.maxi, self.mini]}
        
        else: 
            train_sample_list,test_sample_list = [], []
            test_list, train_list = self.ge_val_list(val_idx)
            print('val{}  val_list: {}'.format(val_idx,test_list))
            

            for flag in flags:
                if flag == 'Normal':
                    continue
                else:
                    print('flag: {}'.format(flag))
                    if len(self.samples_list[flag]) < self.k:
                        para = 'flag{} \n only with {} range \n'.format(flag,len(self.samples_list[flag]))
                        self.paras += para
                        all_list = []
                        for r_range in self.samples_list[flag]:
                            all_list.extend(self.samples_list[flag][r_range])
                        test_sample_list = all_list[val_idx*len(all_list)//self.k:(val_idx+1)*len(all_list)//self.k]
                        train_sample_list = [ _ for _ in all_list if _ not in test_sample_list]
                        print(test_sample_list,train_sample_list)

                    else:
                        para = 'flag{} \ntrain range:{}\ntest range:{}\n'.format(flag,train_list,test_list)
                        self.paras += para
                    
                        r_ranges = os.listdir(os.path.join(self.data_dir, flag, str(self.ab_range[0] - self.ab_range[1])))
                        r_ranges.remove('100')
                        r_ranges.sort()
                        for r_range in tqdm(r_ranges, desc='Generating flag ({}) training and testing index '.format(flag)): 
                            #generate testing data
                            #print(r_range)
                            if r_range in test_list[flag]:                    
                                #print('flag: r_range{}'.format(self.samples_list[flag][r_range]))

                                test_sample_list.extend(self.samples_list[flag][r_range])
                            else:
                                train_sample_list.extend(self.samples_list[flag][r_range])
                        
            if self.normal_use:
                print('Generating Normal training and testing index')
                random.shuffle(self.samples_list['Normal'])
                k_samples = math.ceil(len(self.samples_list['Normal'])//self.k)
                end = k_samples*(val_idx+1)
                if k_samples*(val_idx+1) > len(self.samples_list['Normal']):
                    end = len(self.samples_list['Normal'])
                test_sample_list.extend( self.samples_list['Normal'][k_samples*val_idx:end])
                train_sample_list.extend(self.samples_list['Normal'][0:k_samples*val_idx])
                train_sample_list.extend(self.samples_list['Normal'][end:])
                   
        random.shuffle(train_sample_list)
        random.shuffle(test_sample_list)
        #print('test list: {}'.format(test_sample_list))
        

        return {'train': train_sample_list,
                'test': test_sample_list,
                'maximini':[self.maxi, self.mini]}

    def read_val_list(self, val_idx): #read the list file for validation
        m_names = os.listdir(os.path.join(self.save_dir,str(val_idx)))
        val_list_file = os.path.join(self.save_dir,str(val_idx),m_names[0],'val_list.txt')
        
        with open(val_list_file,'r') as f:
            line = f.readlines()[-1]
            val_lists = line.strip('\n').strip(']').strip('[').replace("'",'').replace(" ",'').split(',')
        
        return val_lists

class TorchDataset(Dataset):
    def __init__(self, file_list, file_dir, time_steps, repeat=1):
        '''
        :param filelist: id for different file
        :param file_dir: divided file samples: saved_dir + time_stpes
        :param repeat: repeat times
        '''
        self.file_list = file_list #list of the train or test files 
        self.time_steps = time_steps
        self.file_dir = os.path.join(file_dir,str(time_steps)) #file saved path
        #self.maxi = np.load(os.path.join(self.file_dir.strip('/data'),'maxi.npy'))
        #self.mini = np.load(os.path.join(self.file_dir.strip('/data'),'mini.npy'))
        self.repeat = repeat
        self.len = self.__len__()


    def __getitem__(self, i):
        index = i % self.len
        # print("i={},index={}".format(i, index))
        file_path = os.path.join(self.file_dir,'data',self.file_list[i]) #transfer to tensor
        all_data = np.load(file_path)
        label =torch.tensor(all_data[-1]) # all_data[:,-1] #channel dimension is in the first dimension
        data = torch.tensor(all_data[:-1]) 

        return data, label
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.file_list) * self.repeat
        return data_len
 


if __name__ == '__main__':

    data_dir = '/mnt/data/datasets/ACCORD'
    fault_flags1 = ['12_FLW4_1', '14_TMP4_1', '21_DBPSCR', '29_VLVBPS', '30_VLVACL', '31_GCI', '32_34_GC1P1_3', '35_GC2P', '38_WPMP1', '41_VVPWC1', '42_VVPWC2']
    class_list1 = ['12_FLW4_1', '14_TMP4_1', '21_DBPSCR', '29_VLVBPS', '30_VLVACL', '31_GCI', '32_34_GC1P1_3', '35_GC2P', '38_WPMP1', '41_VVPWC1', '42_VVPWC2']
    fault_flags = [  '12_FLW4_1', '31_GCI']
    class_list = [ '12_FLW4_1']
    fault_flags1 = ['12_FLW4_1', '29_VLVBPS', '30_VLVACL', '31_GCI', '32_34_GC1P1_3', '35_GC2P', '38_WPMP1', '41_VVPWC1', '42_VVPWC2']
    fault_flags = ['29_VLVBPS']
    save_dir = '/mnt/database/torch_results/100VERIFY'
    ab_range = [1061, 1001]
    time_steps = 400
    channels = 76
    k = 10
    normal_use = True 
    ratio = 0.7
    sli_time = 3000
    val_idx = 0
    threshold_data = 0.0
    if normal_use:
        fault_flags.append('Normal')
    snr = 10
    channel_snr = None
    noise_type = 'Guassian'
    load_filelist=False

    dataset = ACCORDDataset(data_dir=data_dir, 
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
                            noise_type=noise_type,normal_increase=True)
    #filelist = dataset.read_test_file()
    #print(filelist)
    #val_dir = dataset.save_file_list(names=['cnn'],k=10)
    #print(val_dir)
    #val_list_read = dataset.read_val_list(9)
    
    #val_list = dataset.ge_val_list(9)
    #print(set(val_list_read))
    #print(set(val_list_read).difference(set(val_list)))
    #assert val_list_read == val_list
    datasets = dataset.generate_datasets(fault_flags,9)
    #print(dataset.r_ranges)
    print(datasets['test']) #return the test list filename of the numpy file

    print(datasets['maximini'][0],datasets['maximini'][1])
    
    dataloader = DataLoader(dataset=TorchDataset(file_list=datasets['train'],file_dir=save_dir,time_steps=time_steps), batch_size=8)
    channels = dataset.get_channels() 
    print(channels)
    labels = None
    for data,label in tqdm(dataloader):
        print('label',label, label.size())
        print('data',data, data.size())

        #print('label',data[1])






