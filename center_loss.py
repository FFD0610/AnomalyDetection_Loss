import torch.nn as nn
import torch
import torch.nn.functional as F
  

'''
Using the MSE to calculate the loss
'''
class CenterLoss(nn.Module):
    def __init__(self,  feat_dim, classes,  alpha=0.5):
        super(CenterLoss, self).__init__()
        self.classes = classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(self.classes, feat_dim))

    def forward(self, data, labels, model ):
        '''
                Args:
            x: feature matrix with shape (batch_size,  feat_dim, time_Step,).
            labels: ground truth labels with shape (batch_size,time step).
        '''
        device = (torch.device('cuda')
                  if data.is_cuda
                  else torch.device('cpu'))
        feature, cls = model(data)

        feature = feature.transpose(2,1).reshape(-1,feature.size()[1]) #flatten into the 1-dimension
        feature = F.normalize(feature.squeeze(-1), p=2, dim=1)
        #self.centers = F.normalize(self.centers.squeeze(-1), p=2, dim=1)
        orininal_labels = labels.type(torch.LongTensor)
        labels = labels.type(torch.LongTensor).flatten() #flatten into the 1-dimension

        centers_batch = self.centers.index_select(dim=0, index=labels).to(device)#.long()) ensure each center for each batch data
        center_loss = nn.MSELoss().to(device)(feature, centers_batch)

        # Calculate the difference
        diff = centers_batch - feature
        unique_labels, unique_indices = torch.unique(labels, return_inverse=True) 
        #find out the unqie label within dataset: delete the repeat samples
        #unique label and unique index in the new labels it output from the unqie label
        #unique indices are the index of the which label
        difference = torch.zeros_like(self.centers).to(device)
        for i in unique_labels:
            mask = (labels == i) 
            difference[i] += torch.sum(diff[mask],dim=0)
            #difference[i] += diff[j]
        appear_times = torch.bincount(labels,minlength=self.classes).to(device)
        difference /= appear_times.clamp(min=1).unsqueeze(1)

        assert difference.size() == self.centers.size()
        return center_loss, difference, cls, orininal_labels.to(device)


    def update_centers(self, diff):
        with torch.no_grad():
            self.centers += -self.alpha * diff.to(torch.device('cuda') if self.centers.is_cuda else torch.device('cpu'))

'''
Using the classification distance from each center to calculate the loss
Return one more distance 
'''
class CenterLoss2(nn.Module):
    #use the distance as the classification results 
    #for 3 dimension dataset input into the model
    def __init__(self,  feat_dim, subj_num,  alpha=0.5):
        super(CenterLoss2, self).__init__()
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
        
        #feature = feature.transpose(2,1).reshape(-1,feature.size()[1]) #flatten into the 1-dimension 9.2 code test 
        feature = feature.reshape(-1,feature.size()[1])
        label = label.flatten() #flatten into the 1-dimension
        batch_size = feature.size(0)

        expanded_centers = self.centers.expand(batch_size, -1, -1) # for all batch

        expanded_feature = feature.expand(self.num_classes, -1, -1).transpose(1, 0) #for each type of the center using the samples for each center??
        distance_centers = (expanded_feature - expanded_centers).pow(2).sum(dim=-1)
 
        
        centers_batch = self.centers.index_select(dim=0, index=label)#.long()) ensure each center for each batch data
        criterion = nn.MSELoss()
        center_loss = criterion(feature, centers_batch)
        
        # Calculate the difference
        diff = centers_batch - feature
        unique_labels, _ = torch.unique(label, return_inverse=True) #find out the unqie label within dataset: delete the repeat samples

        difference = torch.zeros_like(self.centers,)
        for i in unique_labels:
            mask = (label == i) 
            difference[i] += torch.sum(diff[mask],dim=0)

        appear_times = torch.bincount(label,minlength=self.num_classes)
        difference /= appear_times.clamp(min=1).unsqueeze(1)

        assert difference.size() == self.centers.size()

        return center_loss, difference, distance_centers

    def update_centers(self, diff, label):
        with torch.no_grad():
            #self.centers.index_add_(0, label, -self.alpha * diff)
            self.centers += -self.alpha * diff




    
if __name__ == '__main__':
    x = torch.randn((4,2,5)) #(B,features,time steps) #what if to the two dimension task 
    x = torch.arange(4).view(-1,1)
    x = torch.cat((x.repeat(1,5).view(4,1,5),x.repeat(1,5).view(4,1,5)),dim=1)
    #x = torch.ones((4,2,5))*torch.tensor(0,1,2,3) #
    y = torch.randint(high=6,size=(8,1))
    y = torch.arange(4).view(-1,1)
    loss = SupConLoss()
    results = loss(x,y)
    print(results)



