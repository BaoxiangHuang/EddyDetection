# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 14:57:30 2022

@author: 肖肖
"""

import os
import json
import numpy as np
import netCDF4 as nc
from tqdm import tqdm


#(1)切割sst

main_dir='I:/SST/sst'

#返回一个包含所有****年nc文件的路径列表
def get_file(main_dir):
    nc_path=[]
    for root,dirs,files in os.walk(main_dir):
      for file in files:
          if file.endswith(".nc"):
              nc_path.append(os.path.join(root, file))
                                                          
    return nc_path


nc_path=get_file(main_dir)


#获取数据
def get_data(nc_path):
    sst=[]
    for i in range(0,len(nc_path)):
         dataset=nc.Dataset(nc_path[i])
         sst.append(np.array(dataset.variables['sst']))
    
    return sst

nc_sst=list()
nc_sst=get_data(nc_path)

for p in tqdm(range(len(nc_sst))):
        year=nc_path[p][-7:-3]
        z=nc_sst[p].shape[0]
        sst=np.zeros((z,224,224))
        for i in range(z):
          for j in range(360,584):
             for k in range(480,704):
                if nc_sst[p][i][j][k]=='nan' or nc_sst[p][i][j][k]>50 or nc_sst[p][i][j][k]<-20:
                     sst[i][j-360][k-480]=-20
                else:
                     sst[i][j-360][k-480]=nc_sst[p][i][j][k]
    
        np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/sst/'+year+'_sst.npy',sst)

 

 #(2)  切割sla   
year_list=['2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018']  
main_dir='I:/SLA/sla'

#返回一个包含所有****年nc文件的路径列表
def get_file(main_dir):
    nc_path=[]
    for root,dirs,files in os.walk(main_dir):
      for file in files:
          if file.endswith(".nc"):
              nc_path.append(os.path.join(root, file))
              #nc_path=os.path.join(root, file)                                             
    return nc_path


#获取数据
def get_data(nc_path):
    sla=[]
    for i in range(0,len(nc_path)):
         dataset=nc.Dataset(nc_path[i])
         sla.append(np.array(dataset.variables['sla']))
    
    return sla

for i in tqdm(range(len(year_list))):
    year_dir=main_dir+'/'+year_list[i]
    nc_path=get_file(year_dir)#一年
    
    nc_sla=list()
    nc_sla=get_data(nc_path)
    z=len(nc_sla)
    sla=np.zeros((z,224,224))
    for m in range(len(nc_sla)):
         for j in range(360,584):
             for k in range(480,704):
                if nc_sla[m][0][j][k]=='nan' or nc_sla[m][0][j][k]>10 or nc_sla[m][0][j][k]<-10:
                     sla[m][j-360][k-480]=-10
                else:
                     sla[m][j-360][k-480]=nc_sla[m][0][j][k]
                     
    np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/sla/'+year_list[i]+'_sla.npy',sla)


# （2）切割label数据
main_dir = 'F:/eddy_identification and detection/two_class_model/new_eddy_two_label/use'


# 返回一个包含所有****年nc文件的路径列表
def get_file(main_dir):
    npy_path = []
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            if file.endswith(".npy"):
                npy_path.append(os.path.join(root, file))
                # nc_path=os.path.join(root, file)
    return npy_path


npy_path = get_file(main_dir)

for p in tqdm(range(len(npy_path))):

    x = np.load(npy_path[p], allow_pickle=True)
    year = npy_path[p][-14:-10]
    z = x.shape[0]
    label = np.zeros((z, 224, 224))
    for i in range(z):
        for j in range(360, 584):
            for k in range(480, 704):
                label[i][j - 360][k - 480] = x[i][j][k]

    np.save('F:/eddy_identification and detection/two_class_model/dataset224-KE/label/' + year + '_label.npy',
            label)


