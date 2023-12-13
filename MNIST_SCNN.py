#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
from SConv import SConv_2d

class ConvNet(nn.Module):
    def __init__(self, Coefficient_7, Coefficient_3):
        super(ConvNet, self).__init__()
        
        self.Coefficient_7 = Coefficient_7
        self.Coefficient_3 = Coefficient_3
        
        # conv11
        self.conv11 = SConv_2d(in_channels=1,out_channels=32, kernel_size=7, padding=3)
        self.bn11 = nn.BatchNorm2d(32)
        # conv12
        self.conv12 = SConv_2d(in_channels=32,out_channels=32,kernel_size=7, padding=3)
        self.bn12 = nn.BatchNorm2d(32)
        self.mp12 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #conv21
        self.conv21 = SConv_2d(in_channels=32,out_channels=64,kernel_size=7, padding=3)
        self.bn21 = nn.BatchNorm2d(64)
        # conv22
        self.conv22 = SConv_2d(in_channels=64,out_channels=64,kernel_size=7, padding=3)
        self.bn22 = nn.BatchNorm2d(64)
        self.mp22 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # conv31
        self.conv31 = SConv_2d(in_channels=64,out_channels=128,kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(128)
        # conv32
        self.conv32 = SConv_2d(in_channels=128,out_channels=128,kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(128)
        self.ap32 = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)       
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        
        x = self.conv11(x, self.Coefficient_7)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.conv12(x, self.Coefficient_7)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.mp12(x)
        
        x = self.conv21(x, self.Coefficient_7)
        x = self.bn21(x)
        x = F.relu(x)
        x = self.conv22(x, self.Coefficient_7)
        x = self.bn22(x)
        x = F.relu(x)
        x = self.mp22(x)
        
        x = self.conv31(x, self.Coefficient_3)
        x = self.bn31(x)
        x = F.relu(x)
        x = self.conv32(x, self.Coefficient_3)
        x = self.bn32(x)
        x = F.relu(x)       
        
        x = self.ap32(x)
        x = self.fc(x.view(x.size()[:2]))
        #x = F.softmax(x)
               
        return x
        
def get_cnn(Coefficient_7, Coefficient_3):
    return ConvNet(Coefficient_7, Coefficient_3)

