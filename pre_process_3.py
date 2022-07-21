# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 10:16:07 2022

@author: Huang Baoxiang
"""
import numpy as np
#import h5py
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random

#按通道 拼接训练集中的SST,SLA数据
train_sst_path='F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/train_sst_nms.npy'
train_sla_path='F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/train_sla_nms.npy'

train_sst=np.load(train_sst_path,allow_pickle=True)
train_sla=np.load(train_sla_path,allow_pickle=True)

mix0=np.zeros((1,2,224,224))
for j in range(train_sst.shape[0]):
     t=train_sst[j][np.newaxis,:,:]
     s=train_sla[j][np.newaxis,:,:]
     temp=np.concatenate((t,s),axis=0)#(2,128,128)
     temp=temp[np.newaxis,:,:,:]#(1,2,128,128)
     mix0=np.concatenate((mix0,temp),axis=0)
mix0=np.delete(mix0,0,0) #（366,2,128,128）
print(mix0.shape)
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/train_mix_nms.npy',mix0)

#按通道 拼接测试集中的SST,SLA数据
test_sst_path='F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/test_sst_nms.npy'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
test_sla_path='F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/test_sla_nms.npy'

test_sst=np.load(test_sst_path,allow_pickle=True)
test_sla=np.load(test_sla_path,allow_pickle=True)

mix1=np.zeros((1,2,224,224))
for j in range(test_sst.shape[0]):
     t=test_sst[j][np.newaxis,:,:]
     s=test_sla[j][np.newaxis,:,:]
     temp=np.concatenate((t,s),axis=0)#(2,128,128)
     temp=temp[np.newaxis,:,:,:]#(1,2,128,128)
     mix1=np.concatenate((mix1,temp),axis=0)
mix1=np.delete(mix1,0,0) #（366,2,128,128）

print(mix1.shape)
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/test_mix_nms.npy',mix1)




