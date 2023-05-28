import torch
import torch.nn as nn
import torch.nn.functional as F



class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight
    
    
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
       # print({'PSA前':x.shape})
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
       # print({'PSA后':out.shape})
        return out









class HFR_module(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #PSAModule(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
          #  PSAModule(out_channels, out_channels)
            
            
        )
        
        

    def forward(self, x):
        return self.double_conv(x)


class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(2),
            HFR_module(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(2),
            HFR_module(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(2),
            HFR_module(in_channels, out_channels),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
    


class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2)
         

        self.conv = nn.Sequential(
            HFR_module(in_channels, out_channels),
            nn.Dropout2d(p=0.4))
        self.maxpool2=nn.MaxPool2d(2)
        self.maxpool4=nn.MaxPool2d(4)
       # self.changeC=nn.Conv2d(in_channels, out_channels, kernel_size)
       


    def forward(self, x1, x2,x3,x4):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        
       # x = torch.cat([x2, x1], dim=1)
        x3=self.maxpool2(x3)
        x4=self.maxpool4(x4)
        x = torch.cat([x4, x3], dim=1)
        x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, x1], dim=1)
        return self.conv(x)
 


    
    
class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
           self.up = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2)
              

        self.conv = nn.Sequential(
            HFR_module(in_channels, out_channels),
            nn.Dropout2d(p=0.3))

        self.maxpool2=nn.MaxPool2d(2)
        


    def forward(self, x1, x2,x3,x4):
 
        x1 = self.up(x1)

        # input is CHW
        diffY = torch.tensor([x3.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x3.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        x2=self.up(x2)

        x4=self.maxpool2(x4)

        x = torch.cat([x4, x3], dim=1)
        x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, x1], dim=1)

        return self.conv(x)


class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
           self.up = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=2, stride=2)
      
       
        self.up4=nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=4, stride=4)
        self.conv = nn.Sequential(
            HFR_module(in_channels, out_channels),
            nn.Dropout2d(p=0.2))
        
        

    def forward(self, x1, x2,x3,x4):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x4.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x4.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x2=self.up4(x2)
        x3=self.up(x3)
        x = torch.cat([x4, x3], dim=1)
        x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, x1], dim=1)
        
       
        return self.conv(x)
  
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)