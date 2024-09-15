import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter
import numpy as np
#instance refer to the in one samples, it can include multiple conditions at the same time

def checkId(target, a):
    b = []
    for index, nums in enumerate(a):
        if nums == target:
            b.append(index)
    return (b) #set belongs to one class
#nb__trans: the number of the window from each time series dataset --> get how many piece of the comparison data from the setteld:
#2/3 
def MixedLoss(nb_trans, z, labels, alpha=0.34, beta=0.33, temporal_unit=0):
    #z --> features (in the format of the B*  nb_trans, time_Steps, channels)
    #nb_trans --> used for comparsion samples
    loss = torch.tensor(0., device=z.device)
    labels = labels.to(device=z.device)
    d = 0
    while z.size(1) > 1:
        if alpha != 0:
            inter = alpha * supervised_contrastive_loss_inter(z, labels, nb_trans)
            loss += inter
            
        if beta != 0:
            intra = beta * supervised_contrastive_loss_intra(z, labels, nb_trans)
            loss += intra

        if d >=temporal_unit:
            if 1 - alpha - beta != 0:
                loss += (1 - alpha - beta) * self_supervised_contrastive_loss(z, nb_trans)
        d += 1
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2) #num --> 7
    #use the pooling layer for multi-scaled time series analysis
    if z.size(1) == 1:
        if alpha != 0:
            inter = alpha * supervised_contrastive_loss_inter(z, labels, nb_trans)
            loss += inter
            
        if beta != 0:
            intra += beta * supervised_contrastive_loss_intra(z, labels, nb_trans)
            loss += intra
        d += 1
    return loss / d #, inter. intra
    #use the pooling layer to decrease the dimension of the dataset to pause the iteration recursively

def supervised_contrastive_loss_inter(z, labels, nb_trans):
    #supervised contrasive learning for inter classification 
    #inter: among different groups --> Supervised contrasive learning 
    #z: vector for the input
    labels = labels.contiguous().view(-1, 1)
    logits_mask = torch.eq(labels, labels.T).float()
    logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1] #return the lower triangular part of the matrix --> -1: diagonal was set to 0
    logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:] 
    #do not include the last and first channles? why --> set to zero in upper triangular means the final channels becomes to 0 
    #but for the first dimension, why do not include the first diagonal value
    B, T = z.size(0)/nb_trans, z.size(1) #why B is divided by the trans --> they are saved in the concated 0 dimension
    if B == 1:
        return z.new_tensor(0.)
    z = z.transpose(0, 1)  # T x nb_transB x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x nb_transB x nb_transB
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x nb_transB x (nb_transB-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:] # T x nb_transB x (nb_transB-1) --> #(nb_transB-1)x (nb_transB-1)?
    #except the diagonal value 
    logits = -F.log_softmax(logits, dim=-1)#log+softmax --> T x nb_transB x (nb_transB-1)
    logits = logits*logits_labels
    logits_ave = torch.sum(logits_labels, dim=1) #calculate the samples in the positive 
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
    return loss

def supervised_contrastive_loss_intra(z, labels, nb_trans):
    #the distance within one group 
    #the positive samples only comes from the same type time steps one 
    B, T = z.size(0)/nb_trans, z.size(1)
    if B == 1:
        return z.new_tensor(0.)
    labels_list = labels
    labels_list = labels_list.squeeze().tolist()
    count = Counter(labels_list) #count each values and its presented times
    set_count = set(count)
    class_of_labels = len(set_count)
    if class_of_labels == B:
        return z.new_tensor(0.)
    loss_sum = torch.tensor(0., device=z.device)
    i = 0
    for key in count:
        #for different class 
        index_label = checkId(key, labels) #search out each element with its index in one class
        data_key = z[index_label]  #search out the same class data
        data_key = data_key.to(device=z.device)
        nb_initial = data_key.size(0)/nb_trans
        if nb_initial == 1: #? for the last samples
            loss_sum += 0
            i +=1
            break
        temperal_label = torch.arange(0, nb_initial) #add the new labels through the time it appears
        temperal_label = temperal_label.to(device=z.device)
        temperal_label = temperal_label.repeat(nb_trans)#repeat it m times with trans labels for each time series data samples
        temperal_label = temperal_label.contiguous().view(-1, 1)
        logits_mask = torch.eq(temperal_label, temperal_label.T).float()
        logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
        logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:]
        data_key = data_key.transpose(0, 1)  # T x nb_initial*nb_trans x C #data_key: the value subjected into one sample
        sim = torch.matmul(data_key, data_key.transpose(1, 2))  # T x nb_initial*nb_trans x nb_initial*nb_trans
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x nb_initial*nb_trans x (nb_initial*nb_trans-1) 
        #---> this can definitely select the whole triangel matrix without the diagonal
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]  # T x nb_initial*nb_trans x (nb_initial*nb_trans-1)
        #delete the first and last line 
        logits = -F.log_softmax(logits, dim=-1)
        logits = logits * logits_labels #label all is 1 for the same class
        logits_ave = torch.sum(logits_labels, dim=1) #Count what samples in one class
        loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
        loss_sum += loss
        i +=1
    loss_sum = loss_sum/i #for different classification, calculating the different time step samples
    return loss_sum

def self_supervised_contrastive_loss(z, nb_trans):
    #without label information --> doing the self-contrasive by itself
    #z: 12 7 3 
    B, T = int(z.size(0)/nb_trans), z.size(1)
    if T == 1:
        return z.new_tensor(0.)
    temperal_label = torch.arange(0, T) #why arrange to the T ^^> for temporal inforamtion
    # for each input: B T C -> flatten the BT:  the amount of labels is needed increase
    temperal_label = temperal_label.to(device=z.device)
    temperal_label = temperal_label.repeat(nb_trans)
    temperal_label = temperal_label.contiguous().view(-1, 1)
    logits_mask = torch.eq(temperal_label, temperal_label.T).float()
    logits_labels = torch.tril(logits_mask, diagonal=-1)[:, :-1]
    logits_labels += torch.triu(logits_mask, diagonal=1)[:, 1:] #if it subect to the same time step or not 

    z = z.reshape(B, T*nb_trans, z.size(-1)) #apart the trans into the second dimension
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1) # 4, 21 ,10 <-- nb_trans
    logits = logits * logits_labels
    logits_ave = torch.sum(logits_labels, dim=1) #only preserve the one not in the same time steps
    loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
    return loss



def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    #index increase the dimension and was filled with the np.arrange parts
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx].transpose(2,1)
        #  A [index.reshape(-1,1),(index.shape[0],num_elem]]
#all_index is used to increase the second dimension for the all_index indexed
#final return the B * num_elem * channels 

class Mixed_calculation(nn.Module):
    def __init__(self,  alpha=0.34, beta=0.33, temporal_unit=0):
        super(Mixed_calculation, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temporal_unit = temporal_unit

    def forward(self, data, labels, model):
        device = (torch.device('cuda')
                  if data.is_cuda
                  else torch.device('cpu'))
        
        labels = labels.mean(dim=-1,keepdim=True).type(torch.LongTensor).to(device)
        #model output: (B,C,T)
        ts_l = data.size()[-1]#'length of the time series'
        temporal_unit = 1
        crop_l = np.random.randint(2 ** (temporal_unit + 1), high=ts_l+1) #crepped length 6
        crop_left = np.random.randint(ts_l - crop_l + 1) #crepped left: left position for generating the value 7
        crop_right = crop_left + crop_l#crepped right 13
        nu_trans = np.random.randint(3, 4)
        crop_eleft_1 = np.random.randint(crop_left + 1)
        crop_eright_1 = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_eleft_2 = np.random.randint(crop_eleft_1 + 1)
        crop_eright_2 = np.random.randint(low=crop_eright_1, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft_2, high=ts_l - crop_eright_2 + 1, size=data.size(0))
        #take_per_row output: B C T 

        out1, class1 = model(take_per_row(data.transpose(1,2), crop_offset + crop_eleft_2, crop_right - crop_eleft_2))
        out1 = out1.transpose(2,1)[:, -crop_l:]
        class1 = class1[:, :, -crop_l:]
        
        out2, class2 = model(take_per_row(data.transpose(1,2), crop_offset + crop_eleft_1, crop_eright_1 - crop_eleft_1))
        out2 = out2.transpose(2,1)[:, (crop_left-crop_eleft_1):(crop_right-crop_eleft_1)]
        class2 = class2[:, :, (crop_left-crop_eleft_1):(crop_right-crop_eleft_1)]

        out3, class3 = model(take_per_row(data.transpose(1,2), crop_offset + crop_left, crop_eright_2 - crop_left))
        out3 = out3.transpose(2,1)[:, :crop_l]
        class3 = class3[:, :, :crop_l]

        output = torch.cat([out1, out2, out3], 0) # B, T, C
        sup_labels = labels.repeat(3,1)

        loss_h = MixedLoss(
            nu_trans,
            output,
            sup_labels,
            temporal_unit=temporal_unit
        )
        labels_out = labels.repeat(3,class1.size()[-1]).type(torch.LongTensor).to(device)
        classfication = torch.cat([class1, class2, class3], 0)
        return loss_h, output, classfication, labels_out



