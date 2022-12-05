import torch
import torch.nn as nn


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1, mode='sum'):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight
        self.mode = mode

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()

        if self.mode == 'mean':
            return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
        elif self.mode == 'sum':
            return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)