#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class SConv_2d(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, same=False):
        super(SConv_2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same
        self.conv = nn.Conv2d(in_channels, out_channels, self.k[0], padding=0, stride=self.k[0])

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    
    def forward(self, x, Coefficient):
              
        x = F.pad(x, self._padding(x), mode='constant',value=0)
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        B, C, H, W, KH, KW = x.shape
        x = x.contiguous().view(B, C, H * W, KH * KW).permute(0, 3, 1, 2) #[#B, KH*KW, C, H*W]
        
        #3x3
        if self.k[0]==3: 
            
            center = x[:,4,:,:]
            center = center.view(B, 1, C, H * W)
            
            #######################################################
            #x: [b, 9, C, H*W]
            #Coefficent: [9, 9]
            #y: [b, 9, C, H*W]
            
            x = torch.einsum('bjcn, ij -> bicn', x, Coefficient)
            
            x = x[:,torch.arange(x.size(1))!=4,:,:]
            x, indices = torch.sort(x, dim=1)
            
            x = torch.cat((x,center),1)
            x = x[:,[0,1,2,3,8,4,5,6,7],:,:]
        
        #5x5
        if self.k[0]==5:
            
            center = x[:,12,:,:]
            center = center.view(B, 1, C, H * W)
            
            #######################################################
            #x: [b, 25, C, H*W]
            #Coefficent: [25, 25]
            
            x = torch.einsum('bjcn, ij -> bicn', x, Coefficient)
            
            x1 = x[:,[6,7,8,11,13,16,17,18],:,:]
            x1, indices1 = torch.sort(x1, dim=1)
            x[:,[6,7,8,11,13,16,17,18],:,:] = x1
            
            x2 = x[:,[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24],:,:]
            x2, indices2 = torch.sort(x2, dim=1)
            x[:,[0,1,2,3,4,5,9,10,14,15,19,20,21,22,23,24],:,:] = x2
            
            x = torch.cat((x1,x2,center),1)
            x = x[:,[8,9,10,11,12,13,0,1,2,14,15,3,24,4,16,17,5,6,7,18,19,20,21,22,23],:,:]
            
        #7x7
        if self.k[0]==7:
            
            center = x[:,24,:,:]
            center = center.view(B, 1, C, H * W)
            
            #######################################################
            #x: [b, 49, C, H*W]
            #Coefficent: [49, 49]
            
            x = torch.einsum('bjcn, ij -> bicn', x, Coefficient)
 
            x1 = x[:,[16,17,18,23,25,30,31,32],:,:]
            x1, indices1 = torch.sort(x1, dim=1)
            x[:,[16,17,18,23,25,30,31,32],:,:] = x1
            
            x2 = x[:,[8,9,10,11,12,15,19,22,26,29,33,36,37,38,39,40],:,:]
            x2, indices2 = torch.sort(x2, dim=1)
            x[:,[8,9,10,11,12,15,19,22,26,29,33,36,37,38,39,40],:,:] = x2
            
            x3 = x[:,[0,1,2,3,4,5,6,7,13,14,20,21,27,28,34,35,41,42,43,44,45,46,47,48],:,:]
            x3, indices3 = torch.sort(x3, dim=1)
            x[:,[0,1,2,3,4,5,6,7,13,14,20,21,27,28,34,35,41,42,43,44,45,46,47,48],:,:] = x3
                        
            x = torch.cat((x1,x2,x3,center),1)
            x = x[:,[24,25,26,27,28,29,30,31,8,9,10,11,12,32,33,13,0,1,2,14,34,35,15,3,48,4,16,36,37,17,5,6,7,18,38,39,19,20,21,22,23,40,41,42,43,44,45,46,47],:,:]
                                          
        x = x.contiguous().view(B, KH, KW, C, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous().view(B, C, KH *H, KW * W)
        x = self.conv(x)
        
        return x