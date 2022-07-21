# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 08:55:25 2021

@author: Huang Baoxiang
"""

import glob
import os
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PSA_EDUNet_Model.PSA_EDUNet_Model import PSA_EDUNet
from torch.nn import functional as F

X_test = np.load('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/X_test_nms.npy',allow_pickle=True) 
Y_test = np.load('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/Y_test_nms.npy',allow_pickle=True)


class MyDataSet(Dataset):
    def __init__(self,data,label):
        super(MyDataSet, self).__init__()
        self.x = data
        self.y = label

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]
    
    


def test_net(net, device,batch_size=1, lr=0.0001):
    # 加载训练集
    
    test_loader = DataLoader(dataset=MyDataSet(X_test,Y_test),
                                               batch_size=batch_size, 
                                               shuffle=False)
    criterion=nn.CrossEntropyLoss()
  

    predict=np.zeros((1,224,224))
    last_label=np.zeros((1,224,224))
    accl=0
    for image, label in test_loader:
          
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
     
            pred = net(image)
        
            
            pred = pred.cpu()
            label = label.squeeze(0)
            label=label.cpu()

            pred = pred.squeeze(0)
            pred = F.softmax(pred, dim=0)

            pred = torch.argmax(pred, dim=0)
           
            k=0
            all=pred.shape[0]*pred.shape[1]
            for m in range(pred.shape[0]):
                for n in range(pred.shape[1]):
                    if(pred[m][n]==label[m][n]):
                        k=k+1
            
            accl=accl+k/all
            print(f'acc:{k/all}')
            
            pred=pred[np.newaxis,:,:]
            predict=np.concatenate((predict,pred),axis=0)
            
            label=label[np.newaxis,:,:]
            last_label=np.concatenate((last_label,label),axis=0)
            
            
            
                    
    print(f'avg_acc:{accl/len(test_loader)}')
    

    predict=np.delete(predict,0,0)
    print(predict.shape)
    last_label=np.delete(last_label,0,0)
    print(last_label.shape)
    np.save('record224-KE/sla_nms/test_result.npy',predict)
    np.save('record224-KE/sla_nms/last_label.npy',last_label)
    
    
if __name__ == "__main__":
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PSA_EDUNet(n_channels=2,  n_classes=3)
    net.to(device=device)
    net.load_state_dict(torch.load('record224-KE/best_model.pth', map_location=device))
    net.eval()
    test_net(net, device)