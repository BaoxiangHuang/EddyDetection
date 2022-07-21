from PSA_EDUNet_Model.PSA_EDUNet_Model import PSA_EDUNet
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader




X_train = np.load('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/X_train_nms.npy',allow_pickle=True) 
Y_train = np.load('F:/eddy_identification and detection/two_class_model/dataset224-KE/tt_nms/Y_train_nms.npy',allow_pickle=True)


class MyDataSet(Dataset):
    def __init__(self,data,label):
        super(MyDataSet, self).__init__()
        self.x = data
        self.y = label

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx],self.y[idx]


def train_net(net, device, data_path, epochs=800, batch_size=8, lr=0.0001):
    # 加载训练集
    
    train_loader = DataLoader(dataset=MyDataSet(X_train,Y_train),
                                               batch_size=batch_size, 
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    #criterion = nn.BCEWithLogitsLoss()
    criterion=nn.CrossEntropyLoss()
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=30,factor=0.5)
    
    epoch_loss=list()
    
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        cur_epoch_loss=torch.zeros(1)
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
     
            pred = net(image)
 
            # 计算loss
            loss = criterion(pred, label.long())

            cur_epoch_loss+=loss.item()
          
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'record224-KE/best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
        

        print(f'Epoch:{epoch}; loss:{cur_epoch_loss.item()/len(train_loader)}')
        epoch_loss.append(cur_epoch_loss.item()/len(train_loader))
        scheduler.step(cur_epoch_loss.item()/len(train_loader))

    np.save('record224-KE/epoch_loss.npy',epoch_loss)

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = PSA_EDUNet(n_channels=2,  n_classes=3)
    net.to(device=device)
    data_path = "data/train/"
    train_net(net, device, data_path)
