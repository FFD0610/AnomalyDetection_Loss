import torch
import torch.nn as nn
import torch.nn.functional as F
from network_spatio_temporal import MixedNet, MixedNet2 
from network_propose_contrasive_center import SeCNN1Donlypropose
#using the hypersphere classifer to represent for final classification task 
#ncad use the classifier based on distance to calculate the anoamly score for each 
from networks_torch import se_layer
#version 1:
#introduce the inter-distance into the center loss

class CenterLoss(nn.Module):
    def __init__(self,  feat_dim, subj_num,  alpha=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = subj_num
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(self.num_classes, feat_dim))

    def forward(self, feature, label):
        '''
                Args:
            x: feature matrix with shape (batch_size,  feat_dim, time_Step,).
            labels: ground truth labels with shape (batch_size,time step).
        '''
        feature = feature.transpose(2,1).reshape(-1,feature.size()[1]) #flatten into the 1-dimension
        label = label.flatten() #flatten into the 1-dimension
        centers_batch = self.centers.index_select(dim=0, index=label)#.long()) ensure each center for each batch data
        criterion = nn.MSELoss()
        center_loss = criterion(feature, centers_batch) #eusure the intra-distance

        diff = centers_batch - feature
        unique_labels, unique_indices = torch.unique(label, return_inverse=True) #find out the unqie label within dataset: delete the repeat samples
        #unique label and unique index in the new labels it output from the unqie label
        #unique indices are the index of the which label
        difference = torch.zeros_like(self.centers)
        for i in unique_labels:
            mask = (label == i) 
            difference[i] += torch.sum(diff[mask],dim=0)
            #difference[i] += diff[j]


        expanded_centers = self.centers.expand(feature.size()[0], -1, -1) # for all batch
        expanded_feature = feature.expand(self.num_classes, -1, -1).transpose(1, 0) #for each type of the center using the samples for each center??
        distance_centers = (expanded_feature - expanded_centers).pow(2).sum(dim=-1) #distace for each centers as for each embeddings
        right_distance = diff.pow(2).reshape(-1,1)#.sum(dim=-1) #distace for right centers
        intra_dis = right_distance/(distance_centers-right_distance)

        #inter-distance: distance outside the belonged type

        appear_times = torch.bincount(label,minlength=self.num_classes)
        #appear_times = appear_times.unsqueeze(-1).expand_as(centers_batch)
        #print(appear_times.clamp(min=1))
        difference /= appear_times.clamp(min=1).unsqueeze(1)
        #print(difference,appear_times)
        assert difference.size() == self.centers.size()

        return center_loss, difference

    def update_centers(self, diff, label):
        with torch.no_grad():
            #self.centers.index_add_(0, label, -self.alpha * diff)
            self.centers += -self.alpha * diff


class InterClassLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_intra=1.0):
        super(InterClassLoss, self).__init__()
        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.lambda_intra = lambda_intra  # 类内损失权重
    
    def cosine_distance(self,x1,x2,dim=-1):
        return 1 - F.cosine_similarity(x1, x2, dim=dim)

    def forward(self, features, labels, nums=2):
        #nums: the number of samples used for inter-class calculation
        features = features.transpose(1,2).reshape(-1,features.size()[1])
        batch_size = features.size(0)
        num_classes = self.centers.size(0)
        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels]
        correct_distances = self.cosine_distance(features, centers_batch, dim=1)#torch.norm(features - centers_batch, dim=1)

        #mask = labels.eq(labels.T) #whether it belong to one class or not 
        select_index = torch.randint(0, features.size()[0], nums) #randomly select for the updated features using the inter-class
        # 计算正确距离
        features_s = features[select_index]
        labels_s = labels[select_index]
        distance = self.cosine_distance(features, centers_batch.T)        
        diff = centers_batch - features
        unique_labels, counts = torch.unique(labels, return_counts=True, sorted=True) #find out the unqie label within dataset: delete the repeat samples
        #unique label and unique index in the new labels it output from the unqie label
        #unique indices are the index of the which label
        expanded_centers = self.centers.expand(features.size()[0], -1, -1) # for all batch
        expanded_feature = features.expand(self.num_classes, -1, -1).transpose(1, 0) #for each type of the center using the samples for each center??
        distance_centers =self.cosine_distance(expanded_feature - expanded_centers).sum(dim=-1) #distace for each centers as for each embeddings
        #sum up trough each chanels
        #right_distance = diff.pow(2).reshape(-1,1)#.sum(dim=-1) #distace for right centers
        correct_distances = mask * distance_centers 
        inter_dis = correct_distances/(distance_centers-correct_distances) #without the right one
        correct_distances = correct_distances.sum(dim=1)/counts #get all the distance for each classification
        centers_batch = self.centers.index_select(dim=0, index=labels)#.long()) ensure each center for each batch data
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)
        #for different class 
        #minimize this part to decrease the distance between the right one

        # 计算比值损失 (L_ratio)
        #ratio_loss = correct_distances / incorrect_distances
        
        # 平均化损失
        loss_ratio = inter_dis.mean()
            #difference[i] += diff[j]
        # 计算类内损失 (L_intra)
        intra_class_loss = 0
        
        # 总损失
        total_loss = loss_ratio.mean() + self.lambda_intra * intra_class_loss + center_loss
        return total_loss
    
    def update_centers(self, features, labels, lr=0.05):
        """
        根据损失函数的梯度信息来更新类别中心，同时考虑类内距离。
        """
        features = features.transpose(1,2)
        features = features.reshape(-1,features.size()[-1])
        batch_size = features.size(0)
        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels]
        
        # 计算梯度并更新中心
        for i in torch.randint(0,batch_size,2):
        #for i in range(batch_size):
            # 获取当前样本的embedding和其对应的正确类别中心
            x = features[i]
            center_correct = centers_batch[labels[i]]
            
            # 计算正确类别中心的更新方向（朝向样本移动）
            grad_correct = (center_correct - x) / torch.norm(center_correct - x)
            self.centers[i] = self.centers[i] - 2* lr * grad_correct

            grad_correct = (centers_batch - x.repeat(self.centers.size()[5,1])) / torch.norm(center_correct - x.repeat(self.centers.size()[5,1]) )
            self.centers = self.centers + lr * grad_correct
        
        #only use two samples

            
            # 对其他类别中心进行更新（远离样本）
            for j in range(self.centers.size(0)):
                if j != labels[i]:
                    center_incorrect = self.centers[j]
                    grad_incorrect = (x - center_incorrect) / torch.norm(x - center_incorrect)
                    
                    # 更新错误类别中心，使其远离当前样本
                    self.centers[j] += lr * grad_incorrect
            
            # 更新正确类别中心，使其靠近当前样本
            self.centers[labels[i]] -= lr * grad_correct



class IntraDistanceLoss1(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_intra=0.1):
        super(IntraDistanceLoss1, self).__init__()
        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.lambda_intra = lambda_intra  # 控制类内损失的权重
    
    def forward(self, features, labels):
        batch_size = features.size(0)
        num_classes = self.centers.size(0)
        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels]
        
        # 计算正确距离
        correct_distances = torch.norm(features - centers_batch, dim=1)
        
        # 计算错误距离 (平均错误类别的距离)
        incorrect_distances = torch.zeros(batch_size, device=features.device)
        for i in range(num_classes):
            if i != labels:
                dist_to_other_centers = torch.norm(features - self.centers[i], dim=1)
                incorrect_distances += dist_to_other_centers
        
        incorrect_distances = incorrect_distances / (num_classes - 1)
        
        # 计算 ratio 损失
        ratio_loss = correct_distances / incorrect_distances
        
        # 计算类内聚合损失 (每个样本与其对应类别中心的距离)
        intra_class_loss = correct_distances.mean()
        
        # 总损失 = ratio loss + λ * intra-class loss
        total_loss = ratio_loss.mean() + self.lambda_intra * intra_class_loss
        
        return total_loss
    
    def update_centers(self, features, labels, lr=0.5):
        """
        根据损失函数的梯度信息来更新类别中心。
        """
        batch_size = features.size(0)
        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels]
        
        # 计算梯度并更新中心
        for i in range(batch_size):
            # 获取当前样本的embedding和其对应的正确类别中心
            x = features[i]
            center_correct = centers_batch[i]
            
            # 计算正确类别中心的更新方向（朝向样本移动）
            grad_correct = (center_correct - x) / torch.norm(center_correct - x)
            
            # 对其他类别中心进行更新（远离样本）
            for j in range(self.centers.size(0)):
                if j != labels[i]:
                    center_incorrect = self.centers[j]
                    grad_incorrect = (x - center_incorrect) / torch.norm(x - center_incorrect)
                    
                    # 更新错误类别中心，使其远离当前样本
                    self.centers[j] += lr * grad_incorrect
            
            # 更新正确类别中心，使其靠近当前样本
            self.centers[labels[i]] -= lr * grad_correct

class IntraDistanceLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_intra=0.1):
        super(IntraDistanceLoss, self).__init__()
        # 初始化类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        self.lambda_intra = lambda_intra  # 控制类内损失的权重
        self.alpha = 0.5
        self.num_classes = num_classes

    def cosine_distance(self, x1, x2, dim=-1):
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(0)
        return 1 - F.cosine_similarity(x1, x2, dim=dim).mean(-1)
    
    def forward(self, features, labels):
        features = features.transpose(1,2).reshape(-1,features.size()[1])
        labels = labels.flatten().reshape(-1,1)
        batch_size = features.size(0)
        num_classes = self.centers.size(0)
        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels].squeeze(1)
        
        # 计算正确距离
        correct_distances =  self.cosine_distance(features, centers_batch, dim=1) #[10,3]
        #correct_distances =  torch.norm(features - centers_batch, dim=1) #[10,3]
        
        # 计算错误距离 (平均错误类别的距离)
        #incorrect_distances = torch.zeros(batch_size, device=features.device)
        all_distance = self.cosine_distance(features,self.centers,dim=1)
        incorrect_distances = all_distance - correct_distances

        incorrect_distances = incorrect_distances / (num_classes - 1)
        
        # 计算 ratio 损失
        #print(correct_distances.size(),incorrect_distances.size())
        ratio_loss = correct_distances / incorrect_distances
        
        # 计算类内聚合损失 (每个样本与其对应类别中心的距离)
        intra_class_loss = correct_distances.mean()
        
        # 总损失 = ratio loss + λ * intra-class loss
        total_loss = ratio_loss.mean() + self.lambda_intra * intra_class_loss #Hinge Loss=max(0,1−y⋅f(x))
        
        return total_loss
    
    def update_centers(self, features, labels, nums=2, lr=0.5):
        """
        根据损失函数的梯度信息来更新类别中心。
        """
        features = features.transpose(1,2).reshape(-1,features.size()[1])
        labels = labels.flatten() #.reshape(-1,1)
        batch_size = features.size(0)

        
        # 获取每个样本对应的正确类别中心
        centers_batch = self.centers[labels].squeeze(1)

        #centers_batch = self.centers.index_select(dim=0, index=labels)#.long()) ensure each center for each batch data
        criterion = nn.MSELoss()
        diff = centers_batch - features

        appear_times = torch.bincount(labels,minlength=self.num_classes)
        #appear_times = appear_times.unsqueeze(-1).expand_as(centers_batch)
        #print(appear_times.clamp(min=1))

        unique_labels, unique_indices = torch.unique(labels, return_inverse=True) #find out the unqie label within dataset: delete the repeat samples
        #unique label and unique index in the new labels it output from the unqie label
        #unique indices are the index of the which label
        difference = torch.zeros_like(self.centers)
        for i in unique_labels:
            mask = (labels == i) 
            difference[i] += torch.sum(diff[mask],dim=0)
        difference /= appear_times.clamp(min=1).unsqueeze(1)

        with torch.no_grad():
            #self.centers.index_add_(0, label, -self.alpha * diff)
            self.centers += -self.alpha * difference
            select_index = torch.randint(0, features.size()[0], (nums,)) #randomly select for the updated features using the inter-class
        # 计算正确距离
            features_s = features[select_index]
            labels_s = labels[select_index]
            distance = self.cosine_distance(features, centers_batch,dim=1)  #scalar
        # 计算梯度并更新中心
            for i in range(batch_size):
              if i in select_index:
                # 获取当前样本的embedding和其对应的正确类别中心
                x = features[i]
                center_correct = centers_batch[i]
                
                # 计算正确类别中心的更新方向（朝向样本移动）
                grad_correct = (center_correct - x) / torch.norm(center_correct - x)
                
                # 对其他类别中心进行更新（远离样本）
                for j in range(self.centers.size(0)):
                    #print(labels[i], i, j)
                    if j != labels[i]:
                        center_incorrect = self.centers[j]
                        grad_incorrect = (x - center_incorrect) / torch.norm(x - center_incorrect)
                        
                        # 更新错误类别中心，使其远离当前样本
                        self.centers[j] += lr * grad_incorrect
                
                # 更新正确类别中心，使其靠近当前样本
                #self.centers[labels[i]] -= lr * grad_correct
        


if __name__ == '__main__':
    x = torch.randn((2,3,5)).repeat(5,1,1) #[torch.randn((2,3,5)),torch.randn((2,3,5))]
    labels =  torch.arange(10).reshape(2,5).repeat(5,1)#((2)).reshape(-1,1)
    model = SeCNN1Donlypropose(5,3,10)
    features, cls = model(x)
    loss = IntraDistanceLoss(num_classes=10,feature_dim=128)
    to_loss = loss(features,labels)
    loss.update_centers(features,labels)
    #distance,center_loss, sphere_loss = loss(features,labels)
    # = loss.compute_loss(cls,labels,center_loss,sphere_loss)
    #distance = cosine_similarity(x[0],x[1])
    print(to_loss)