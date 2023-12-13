#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Interpolation_Coefficient
from SConv import SConv_2d


class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        
        super(BasicBlock, self).__init__()
        
        Coefficient_3 = Interpolation_Coefficient(3)
        
        Coefficient_3 = Coefficient_3.cuda()
        
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1 = SConv_2d(Coefficient = Coefficient_3, in_channels=in_planes, out_channels=planes, kernel_size=3, feat_stride=1, conv_stride=stride, padding=1)
        
        self.bn1 = nn.BatchNorm2d(planes)
        
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3， stride=1, padding=1, bias=False)
        self.conv2 = SConv_2d(Coefficient = Coefficient_3, in_channels=planes, out_channels=planes, kernel_size=3, feat_stride=1, conv_stride=1, padding=1)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion * planes:
            
            self.shortcut = nn.Sequential(
                
                #kernel_size = 1
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(x)
        
        out = F.relu(out)
        
        return out
    

class ResNet(nn.Module):
    
    def __init__(self, block, num_blocks, num_classes=24):
        
        super(ResNet, self).__init__()
        
        #block: BasicBlock
        #planes: [3, 4, 6, 3]
        
        Coefficient_3 = Interpolation_Coefficient(3)
        
        Coefficient_3 = Coefficient_3.cuda()
        
        self.in_planes = 64
        
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = SConv_2d(Coefficient = Coefficient_3, in_channels=1, out_channels=64, kernel_size=3, feat_stride=1, conv_stride=1, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias, 0)

                
    def _make_layer(self, block, planes, num_blocks, stride):
        
        #BasicBlock,  64, 3, 1
        #BasicBlock, 128, 4, 2
        #BasicBlock, 256, 6, 2
        #BasicBlock, 512, 3, 2
        
        #[1] + [1]*(3-1) = [1, 1, 1]
        #[2] + [1]*(4-1) = [2, 1, 1, 1]
        #[2] + [1]*(6-1) = [2, 1, 1, 1, 1, 1]
        #[2] + [1]*(3-1) = [2, 1, 1]
        
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            
            layers.append(block(self.in_planes, planes, stride))
            
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, (8,8))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])
