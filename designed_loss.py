import torch
from torch import nn
import torch.nn.functional as F
from collections import Counter
import numpy as np
#intra distance was comfirmed by randomly divided 
from .soft_contrastive import hier_CL_soft, temp_CL_soft, timelag_sigmoid, dup_matrix

def checkId(target, a):
    b = []
    for index, nums in enumerate(a):
        if nums == target:
            b.append(index)
    return (b) #set belongs to one class
#nb__trans: the number of the window from each time series dataset --> get how many piece of the comparison data from the setteld:
#2/3 

def mask_data_time_sync(data, ratio=0.3, mask_value=float('nan'),num=2):
    """
    对 (B, T) 维度随机屏蔽部分时间步，并在所有 C 维同步 mask。

    Args:
        data (torch.Tensor): 形状为 (B, C, T) 的数据
        ratio (float): 屏蔽的比例
        mask_value (float): 设定屏蔽的值（NaN、0 或 inf）
        num: number of the augmentation

    Returns:
        torch.Tensor: 处理后的数据
        torch.Tensor: Mask 掩码（1 表示被 mask，0 表示未 mask）
    """
    B, C, T = data.shape
    num_masked = int(T * ratio) 
    aug_data = [data]
    for i in range(num):
        all_indices = torch.randperm(T)[:num_masked] #generate a index randmly index 
        mask = torch.zeros(T, dtype=torch.bool, device=data.device) #MASK in all False
        mask[all_indices] = True
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, T)  # 扩展到 (B, C, T) 维度
        masked_data = data #.clone()
        masked_data[mask] = mask_value  # 设定 NaN 或 inf
        aug_data.append(masked_data)
    

    return torch.cat(aug_data,dim=0) #, mask

def MixedLoss(self_embed,
            filtered_embed,
            filtered_labels,
            aug_embed,
            temporal_unit, alpha=0.34, beta=0.33):
    #z --> features (in the format of the B*  nb_trans, time_Steps, channels)
    #nb_trans --> used for comparsion samples
    loss = torch.tensor(0., device=self_embed.device)
    d = 0
    z = filtered_embed #b t c 
    while aug_embed.size(1) > 1:
        #every time step within the model
        if alpha != 0 and z.size(0)>0:
            inter = alpha * supervised_contrastive_loss_inter(z, filtered_labels,3)
            loss += inter

        if d >=temporal_unit: #compelete the self-learning after 1 time calculation of supervised learning
            if  beta != 0:
                loss += beta * self_supervised_contrastive_loss(aug_embed, 3)#label the outside 
            if 1 - alpha - beta != 0 and self_embed.size(1) > 1:   
                z1, z2 = self_embed[:self_embed.size(0)//2], self_embed[self_embed.size(0)//2:]
                tau_temp=2
                timelag = timelag_sigmoid(z1.shape[1],tau_temp*(2**d))
                timelag = torch.tensor(timelag, device=z1.device)
                timelag_L, timelag_R = dup_matrix(timelag)
                loss += (1 - alpha - beta) * temp_CL_soft(z1, z2, timelag_L, timelag_R)
                #loss += (1 - alpha - beta) * temp_CL_soft(z1, z2, soft_labels=None, tau_temp=2, lambda_=0.5, temporal_unit=temporal_unit)
                

        d += 1
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2) #num --> 7
        aug_embed = F.max_pool1d(aug_embed.transpose(1, 2), kernel_size=2).transpose(1, 2)
        self_embed = F.max_pool1d(self_embed.transpose(1, 2), kernel_size=2).transpose(1, 2) if self_embed.size(1) > 1 else self_embed

    #use the pooling layer for multi-scaled time series analysis
    if z.size(1) == 1 and z.size(0)>0:
        if alpha != 0:
            inter = alpha * supervised_contrastive_loss_inter(z, filtered_labels,3)
            loss += inter
        '''   
        if beta != 0:
            loss += beta *self_supervised_contrastive_loss(aug_embed, 3)
            #loss += intra
        ''' 
        d += 1
    if torch.isnan(loss):
        print(f'inter para {torch.isnan(z), filtered_labels}')
        print(f'inter para {(z), filtered_labels}')
        print(f'inter: {supervised_contrastive_loss_inter(z, filtered_labels,3)} soft {(1 - alpha - beta) * temp_CL_soft(z1, z2, timelag_L, timelag_R)} self {beta * self_supervised_contrastive_loss(aug_embed, 3)}')
    return loss / d  #, inter. intra
    #use the pooling layer to decrease the dimension of the dataset to pause the iteration recursively

def supervised_contrastive_loss_inter(z, labels, nb_trans): #witin windows
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

class Design_loss(nn.Module):
    def __init__(self,  alpha=0.34, beta=0.33, temporal_unit=1, mask_ratio=0.05, mask=None):
        super(Design_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temporal_unit = temporal_unit
        self.mask_ratio = mask_ratio
        self.mask = mask

    def forward(self, data, labels, model, m=None):
        device = (torch.device('cuda')
                  if data.is_cuda
                  else torch.device('cpu'))
        #pre set the label is already masked as the ratio 
        labels = labels[:,0].reshape(-1,1)
        #model output: (B,C,T)
        ts_l = data.size()[-1]#'length of the time series'
        temporal_unit = self.temporal_unit
        
        #intra-window only for self-contrastive learning, no need labels
        crop_l = np.random.randint(low= 2 ** (self.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=data.size(0))
        if m == 'dlinear' or m == 'timesnet':
            out_all, class_all = model(data)
            out1, class1 = take_per_row(out_all.transpose(1,2), crop_offset + crop_eleft, crop_right - crop_eleft), take_per_row(class_all.transpose(1,2), crop_offset + crop_eleft, crop_right - crop_eleft)
            out2, class2 = take_per_row(out_all.transpose(1,2), crop_offset + crop_left, crop_eright - crop_left), take_per_row(class_all.transpose(1,2), crop_offset + crop_left, crop_eright - crop_left)
        else:
            out1, class1 = model(take_per_row(data.transpose(1,2), crop_offset + crop_eleft, crop_right - crop_eleft))
            out2, class2 = model(take_per_row(data.transpose(1,2), crop_offset + crop_left, crop_eright - crop_left))#add one head used for evaluate whether it belongs to the unknow or not 
        out1,out2 = out1.transpose(2,1)[:, -crop_l:], out2.transpose(2,1)[:, :crop_l] #B, T, C
        class1, class2 = class1[:, :, -crop_l:], class2[:, :, :crop_l] #B,  C, T
        
        self_embed = torch.cat([out1, out2], 0) #B, T, C

        #inter-window for supervised learning 

        #aug_data = mask_data_independent(data, ratio=self.mask_ratio, mask_value=torch('nan'), num=2) #return 3 type of the augmentation
        aug_data = mask_data_time_sync(data, ratio=self.mask_ratio, mask_value=float('nan'), num=2)
        #aug_embed = [model(aug_data[aug_data.size(0)//3*i:aug_data.size(0)//3*(i+1)])[0] for i in range(len(aug_data))]
        #aug_cls = [model(aug_data[aug_data.size(0)//3*i:aug_data.size(0)//3*(i+1)])[1] for i in range(len(aug_data))]
        #aug_embed = torch.cat(aug_embed, dim=0).transpose(2,1) #B T C 
        #aug_cls = torch.cat(aug_cls, dim=0)#B. C. T 
        aug_embed, aug_cls = model(aug_data)
        aug_embed = aug_embed.transpose(2,1) 
        #with label self-supervised learning 

        valid_indices = ~torch.isnan(labels)  # Returns a boolean tensor        
        # Filter data and labels based on valid indices
        filtered_embed = aug_embed[valid_indices.squeeze(1).repeat(3)]
        filtered_cls = aug_cls[valid_indices.squeeze(1).repeat(3)]
        filtered_labels = labels[valid_indices]
        #filtered_labels = filtered_labels.mean(dim=-1,keepdim=True).type(torch.LongTensor).to(device) #(B,1)
        #aug_labels_sup = filtered_labels.repeat(3,class1.size()[-1]).type(torch.LongTensor).to(device)# filtered_labels.repeat(3)#repeat it m times with trans labels for each time series data samples
        filtered_labels = filtered_labels.repeat(3,1).type(torch.LongTensor).to(device)# filtered_labels.repeat(3)
        #without label for self-contrastive learning 
        aug_labels = labels.repeat(3,1).type(torch.LongTensor).to(device)

        loss_h = MixedLoss(
            self_embed,
            filtered_embed,
            filtered_labels,
            aug_embed,
            temporal_unit=temporal_unit
        )

        return loss_h, filtered_embed, aug_cls, aug_labels




def mask_data_independent(data, ratio=0.3, mask_value=float('nan'), num=2):
    """
    独立对 (B, C, T) 维度的每个 (C, T) 位置随机屏蔽部分数据。

    Args:
        data (torch.Tensor): 形状为 (B, C, T) 的数据
        ratio (float): 屏蔽的比例
        mask_value (float): 设定屏蔽的值（NaN、0 或 inf）

    Returns:
        torch.Tensor: 处理后的数据
        torch.Tensor: Mask 掩码（1 表示被 mask，0 表示未 mask）
    """
    B, C, T = data.shape
    aug_data = [data]
    for i in range(num):
        mask = torch.rand(B, C, T) < ratio  
        masked_data = data.clone()
        masked_data[mask] = mask_value  # 设定 NaN 或 inf
        aug_data.append(masked_data)
    
    
    return aug_data


def mask_data_continuous(data, max_mask_ratio=0.3, mask_value=float('nan')):
    """
    随机屏蔽连续的时间片段，模拟缺失数据。

    Args:
        data (torch.Tensor): 形状为 (B, C, T) 的数据
        max_mask_ratio (float): 最大屏蔽比例
        mask_value (float): 设定屏蔽的值（NaN、0 或 inf）

    Returns:
        torch.Tensor: 处理后的数据
        torch.Tensor: Mask 掩码（1 表示被 mask，0 表示未 mask）
    """
    B, C, T = data.shape
    masked_data = data.clone()
    mask = torch.zeros(B, C, T, dtype=torch.bool)  # 初始 mask 设为 False

    for i in range(B):
        mask_len = int(T * torch.rand(1).item() * max_mask_ratio)  # 随机确定 mask 长度
        if mask_len > 0:
            start = torch.randint(0, T - mask_len, (1,)).item()  # 选择起点
            mask[i, :, start:start+mask_len] = True  # 在所有 C 维度 mask 该时间段

    masked_data[mask] = mask_value
    return masked_data, mask

def mask_labels_and_filter(data, labels, ratio=0.8, mask_value=float('nan')):
    """
    随机按 ratio 屏蔽 label，并筛选未屏蔽的样本。

    Args:
        data (torch.Tensor): 形状为 (B, C, T) 的数据
        labels (torch.Tensor): 形状为 (B,) 的标签
        ratio (float): 屏蔽的比例
        mask_value (float): 设定屏蔽的值（NaN 或 inf）

    Returns:
        torch.Tensor: 未被屏蔽的 data
        torch.Tensor: 未被屏蔽的 labels
    """
    B = labels.size()[0]
    num_masked = int(B * ratio) 
    all_indices = torch.randperm(B)[:num_masked] #generate a index randmly index 
    mask = torch.zeros(B, dtype=torch.bool, device=data.device) #MASK in all False
    mask[all_indices] = True
    #mask = torch.rand(B) < ratio  #generate the mask accoring to the value it randomly generated 
    masked_labels = labels.to(torch.float32).clone()
    masked_labels[mask] = mask_value  # 设定 NaN 或 inf 
    
    # 仅选择未被掩码的部分
    valid_idx = ~mask
    filtered_data = data[valid_idx]
    filtered_labels = labels[valid_idx]  # 只返回原始 label，不带 NaN/inf
    return filtered_data, filtered_labels, masked_labels

'''
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
        nb_initial = data_key.size(0)/nb_trans #divide the data into the contrastive format 
        if nb_initial == 1: #? for the last samples
            loss_sum += 0
            i +=1
            break
        temperal_label = torch.arange(0, nb_initial) #add the new labels through the time it appears
        #to divide apart 
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
        logits = logits * logits_labels #label not in the same trans was assumed to be different classes
        logits_ave = torch.sum(logits_labels, dim=1) #Count what samples in one class
        loss = torch.div(torch.sum(logits, dim=-1), logits_ave).mean()
        loss_sum += loss
        i +=1
    loss_sum = loss_sum/i #for different classification, calculating the different time step/stampes samples
    return loss_sum
'''

if __name__ == '__main__':
    # 示例
    B, C, T = 2, 3, 4
    data = torch.randn(B, C, T)
    '''
    masked_data = mask_data_independent(data, ratio=0.3, mask_value=float('nan'), num=3)
    print("原始数据:\n", data)
    print("Mask 掩码:\n")
    print("Mask 后数据:\n", masked_data)

    masked_data, mask = mask_data_time_sync(data, ratio=0.3, mask_value=float('nan'))
    

    print("Mask same 掩码:\n", mask[0].size())  # 显示第一个样本的 mask
    print("Mask same 后数据:\n", masked_data)

    masked_data, mask = mask_data_continuous(data, max_mask_ratio=0.3, mask_value=float('nan'))

    print("Mask leng 掩码:\n", mask[0].size())  # 显示第一个样本的 mask
    print("Mask leng 后数据:\n", masked_data)

    # 示例
    B, C, T = 10, 3, 5  # 假设有 10 个样本，每个样本 (3,5) 维度
    data = torch.randn(B, C, T)
    labels = torch.randint(0, 5, (B,))  # 假设类别为 0-4
    labels = torch.arange(B)
    print(labels.unsqueeze(1).unsqueeze(1).expand(B, C, T),torch.arange(T).unsqueeze(0).unsqueeze(0).expand(B, C, T))
    filtered_data, filtered_labels, masked_labels = mask_labels_and_filter(data, labels, ratio=0.3, mask_value=float('nan'))

    print("原始 Labels:", labels)
    print("屏蔽后的 Labels:", masked_labels)
    print("筛选出的数据 shape:", filtered_data.shape)
    print("筛选出的 Labels:", filtered_labels)
    '''