import torch.nn as nn
import torch
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode 
        #'all'-->supervised contrasive learning 
        #'one'-->contrasive learning (self-supervised)
        self.base_temperature = base_temperature

    def forward(self, data, labels=None, model=None, mask=None):
        #need the dataset shape: B,L,C
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features, cls = model(data)
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        #features = features.transpose(2,1).reshape(-1,1,features.size()[1])
        features = features.transpose(2,1)
        original_labels = labels.type(torch.LongTensor).to(device)
        #labels = labels.flatten()
        labels = labels.mean(dim=-1,keepdim=True).type(torch.LongTensor).to(device) #pooling the samples
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
            #n_views refered to ??
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
            #(only genertate the diagonal feature in the mask)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) #whether the labels are the same to each other or not 
            #mask shape: labels * labels (one factor compared to other vectors) --> B * B
        else:
            mask = mask.float().to(device)

        

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) if features.dim() == 3 else features#delete one dimension

        contrast_count = features.shape[1] #channels-->#feature Dimension???????why not be the batch size samples
        #for the sencond dimension of the features but not the contrast features
        #seperate the vectors through the order and then rebuild a new vector like the shape of the features
        #seperate the input through the number of samples --> recombine it with batch 
        #get a new_B --_ like reshape it to the batch size 
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] #use one sample for positive samples for contrasive in each batch
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), #one for others 
            self.temperature) 
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) #new_B,1 
        #return the biggest value within each batch for all of the classification
        #logits --> for all contrasive results
        logits = anchor_dot_contrast - logits_max.detach() #delete the one of self multiplly
        #utilize the broadcast for calculation and delete the maximum value in it
        #to counstrcut the denominator(x)
        #may be it was aimed at negative
        #cosine similarity huge --> more similar 
        #loss need the smaller cosine similarity 
        #because the loss is calculated as the same class distance / not same class distance
        #therefore, the loss is positive correlation and negative correlation to each  

        # tile mask
        #this count just come from the reshape step and how many the contrast sample it was utilized for the prediction
        mask = mask.repeat(anchor_count, contrast_count) #repeat it in two directions ? why --> because the former reshape and delete the second dimension of the dataset
        #for each anchor it needs to enlarge the comparsion samples at the number of contrast_count 
        #therefore, the mask need to be multiply into twice
        #mask: labels --> B shape B, Features, features # for self-supervised: the mask is the (1,time step)
        # mask-out self-contrast cases --> no comparison between samples and itself
        #the reshaped data is converted into one integer
        #therefore the label can also be intergered copy
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        ) #build a new mask only have zero value at the diagonal index 
        #even copy and repeat, the diagonal value is the one with itself
         
        mask = mask * logits_mask #why not use the diagonal number-->deleted the one to itself

        # compute log_prob except fpr the one with itself
        exp_logits = torch.exp(logits) * logits_mask  #numerator 
        #the one without the results in itself
        #supervised out type: fist calculate the log value and then go avarage
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) #the value shape used for log: 8*1 
        #why use logit - log to represent for the log_prob
        #logits == (log(exp(logits))) and the division in log --> log - log 
        #here logits conclude all the positive samples(which delete the max value of it) --> multiply with the mask it can acquire the final positive one 

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)#refer to the colums 
        #calculate the positive pairs in each columns
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        #ir there is not postive samples in this sample
        p = mask/mask_pos_pairs.reshape(-1,1) #ground truth distribution
        q = exp_logits #contrasive distibution
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs #dominator
                            #new batch and new batch
        #mask: tell the place which the positive samples exists in
        #mask * log_prob) refers to the value of each position -- except for the self-contrasive
        #mean --> all avaraged to this part  

        # loss: include the entropy cross loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, mean_log_prob_pos, cls, original_labels #, mask * exp_logits



    
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



class SeCNN1Dcontrasive(nn.Module):
  def __init__(self, time_steps, channels=20, classes=10, pos =2):
    super(SeCNN1Dcontrasive, self).__init__()
    self.conv1d1 = nn.Conv1d(in_channels=channels, out_channels=128, kernel_size=9, padding=(9 // 2)) #, padding=(9 // 2)
    #self.lstm = nn.LSTM(input_size=12, hidden_size=128, batch_first=True)
    self.relu1 = nn.ReLU()
    self.norm1 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.9)
    self.conv1d2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=9, padding=(9 // 2)) #, padding=(9 // 2)
    self.relu2 = nn.ReLU()
    self.norm2 = nn.BatchNorm1d(num_features=128, eps=1e-05, momentum=0.9)

    self.conv1d3 = nn.Sequential(nn.Conv1d(in_channels=128, out_channels=classes, kernel_size=1),  )
                                 #nn.Softmax(1)) 
   
    #self.pool = nn.AdaptiveAvgPool1d(1) #not sure 
    self.pos = pos
    if pos == 0:
        self.se = se_layer(channels=channels)
    else:
        self.se = se_layer(channels=128)


  def forward(self, x, y):
    if self.pos == 0:
        x =self.se(x)
    x = self.conv1d1(x)
    x = self.relu1(x)
    x = self.norm1(x)
    if self.pos == 1:
        x =self.se(x)
    x = self.conv1d2(x)
    x = self.relu2(x)
    v = self.norm2(x)
    if self.pos == 2:
        v =self.se(v)
    x = self.conv1d3(v)
    #x = self.pool(x)
    #print('output: ', x.size())
    #return F.normalize(self.pool(v).squeeze(-1), p=2, dim=1),F.normalize(x.squeeze(-1), p=2, dim=1) 
    return F.normalize(v.squeeze(-1), p=2, dim=1),F.normalize(x.squeeze(-1), p=2, dim=1) 
  