""" Full assembly of the parts to form the complete network """
"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn.functional as F

from .PSA_EDUNet_Parts import *
import numpy as np


class PSA_EDUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(PSA_EDUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = HFR_module(n_channels, 64)
        self.down1 = Down1(64, 64)
        self.down2 = Down2(64, 64)
        self.down3 = Down3(64, 64)
        
        self.up3 = Up3(256, 64, bilinear)
        self.up2 = Up2(256, 64, bilinear)
        self.up1 = Up1(256, 64, bilinear)
    
        
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print(x1.shape)
        x2 = self.down1(x1)
       # print(x2.shape)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
       # print(x4.shape)
       

    
        x = self.up3(x4, x3,x2,x1)
      #  print(x.shape)
        x = self.up2(x, x3,x2,x1)
       # print(x.shape)
        x = self.up1(x, x3,x2,x1)
       # print(x.shape)
        logits = self.outc(x)
       # print(logits.shape)
        return logits


"""
inp = torch.rand(1,2,224,224)
#x = torch.tensor(x,dtype=torch.double)
net=EddyNet(n_channels=2,n_classes=3)
y=net(inp)
print({'最后输出y':y.shape})
"""