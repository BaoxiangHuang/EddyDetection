# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 08:44:34 2022

@author: Huang Baoxiang
"""
import os
import numpy as np
from tqdm import tqdm

#训练集：拼接04年--17年 sst,sla 数据，并归一化

sst_dir='F:/eddy_identification and detection/two_class_model/dataset224-KE/sst'
sla_dir='F:/eddy_identification and detection/two_class_model/dataset224-KE/sla'


#返回一个包含所有****年npy文件的路径列表
def get_file(main_dir):
    npy_path=[]
    for root,dirs,files in os.walk(main_dir):
      for file in files:
          if file.endswith(".npy"):
              npy_path.append(os.path.join(root, file))
    return npy_path

sst_path=get_file(sst_dir)
sla_path=get_file(sla_dir)

sst0=np.load(sst_path[0],allow_pickle=True)
sla0=np.load(sla_path[0],allow_pickle=True)

for i in range(1,len(sst_path)-1):
    temp=np.load(sst_path[i],allow_pickle=True)
    sst0=np.concatenate((sst0,temp),axis=0)
    
for j in range(1,len(sla_path)-1):
    temp=np.load(sla_path[j],allow_pickle=True)
    sla0=np.concatenate((sla0,temp),axis=0)
    

def get_value(temp,flag):
   x=list()
    
    
    
   if flag==1:#sst
     for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            for k in range(temp.shape[2]):
                if temp[i][j][k]!=-20:
                    x.append(temp[i][j][k])
     x=np.array(x)
     mu=np.average(x)
     sig=np.std(x)
     
   elif flag==2:#sla
     for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            for k in range(temp.shape[2]):
                if temp[i][j][k]!=-10:
                   x.append(temp[i][j][k])
     x=np.array(x)
     mu=np.average(x)
     sig=np.std(x)
     
             
   return mu,sig


sst_m,sst_sig=get_value(sst0, 1)
sla_m,sla_sig=get_value(sla0, 2)

z=sst0.shape[0]
sst_nms=np.zeros((z,224,224))
sla_nms=np.zeros((z,224,224))

for i in tqdm(range(sst0.shape[0])):
   for j in range(sst0.shape[1]):
       for k in range(sst0.shape[2]):
           sst_nms[i][j][k]=(sst0[i][j][k] - sst_m) / sst_sig
           sla_nms[i][j][k]=(sla0[i][j][k] - sla_m) / sla_sig
           
           
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/train_sst_nms.npy',sst_nms)
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/train_sla_nms.npy',sla_nms)


#训练集：18年 sst,sla 数据归一化
#18年
sst1=np.load(sst_path[14],allow_pickle=True)
sla1=np.load(sla_path[14],allow_pickle=True)
y=sst1.shape[0]
test_sst_nms=np.zeros((y,224,224))
test_sla_nms=np.zeros((y,224,224))

for l in range(sst1.shape[0]):
   for m in range(sst1.shape[1]):
       for n in range(sst1.shape[2]):
           test_sst_nms[l][m][n]=(sst1[l][m][n] - sst_m) / sst_sig
           test_sla_nms[l][m][n]=(sla1[l][m][n] - sla_m) / sla_sig
           
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/test_sst_nms.npy',test_sst_nms)
np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/test_sla_nms.npy',test_sla_nms)

