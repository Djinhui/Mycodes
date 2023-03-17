'''Pytorch的中阶API主要包括各种模型层、损失函数、优化器、数据管道等等
也是pytorch的通用建模流程'''

import os
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 

# 一，线性回归模型
#样本数量
n = 400

# 生成测试用数据集
X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0],[-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0 + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

#构建输入数据管道
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)

# 定义模型
model = nn.Linear(2,1) #线性层
model.loss_func = nn.MSELoss()
model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

# 训练模型
def train_step(model,x, y):
    pred = model(x)
    loss = model.loss_func(pred, y)
    loss.backward()
    model.optimizer.step()
    model.optimizer.zero_grad()
    return loss.item()

def train_model(model, epochs):
    for epoch in range(1,epochs+1):
        for x, y in dl:
            loss = train_step(model, x, y)

        if epoch % 10 == 0:
            printbar()
            w = model.state_dict()['weight']
            b = model.state_dict()['bias']
            print(f'epoch={epoch}, loss={loss}')
            print('w=',w)

train_model(model, 1000)

# 二， DNN二分类模型
#正负样本数量
n_positive,n_negative = 2000,2000

#生成正样本, 小圆环分布
r_p = 5.0 + torch.normal(0.0,1.0,size = [n_positive,1]) 
theta_p = 2*np.pi*torch.rand([n_positive,1])
Xp = torch.cat([r_p*torch.cos(theta_p),r_p*torch.sin(theta_p)],axis = 1)
Yp = torch.ones_like(r_p)

#生成负样本, 大圆环分布
r_n = 8.0 + torch.normal(0.0,1.0,size = [n_negative,1]) 
theta_n = 2*np.pi*torch.rand([n_negative,1])
Xn = torch.cat([r_n*torch.cos(theta_n),r_n*torch.sin(theta_n)],axis = 1)
Yn = torch.zeros_like(r_n)

#汇总样本
X = torch.cat([Xp,Xn],axis = 0)
Y = torch.cat([Yp,Yn],axis = 0)

#构建输入数据管道
ds = TensorDataset(X,Y)
dl = DataLoader(ds,batch_size = 10,shuffle=True,num_workers=2)

class DNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2,4)
        self.fc2 = nn.Linear(4,8)
        self.fc3 = nn.Linear(8,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.sigmoid(self.fc3(x))
        return y

    def loss_func(self, y_pred, y_true):
        return nn.BCELoss()(y_pred, y_true)

    def metric_func(self, y_pred, y_true):
        y_pred = torch.where(y_pred > 0.5,\
            torch.ones_like(y_pred,dtype=torch.float32),\
                torch.zeros_like(y_pred, dtype=torch.float32))
        acc = torch.mean(1-torch.abs(y_true - y_pred))
        return acc

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

model = DNNModel()
# 测试模型结构
(features,labels) = next(iter(dl))
predictions = model(features)

loss = model.loss_func(predictions,labels)
metric = model.metric_func(predictions,labels)

print("init loss:",loss.item())
print("init metric:",metric.item())
    

def train_step(model, features, labels):

    # 正向传播求损失
    predictions = model(features)
    loss = model.loss_func(predictions,labels)
    metric = model.metric_func(predictions,labels)

    # 反向传播求梯度
    loss.backward()
    # 更新模型参数
    model.optimizer.step()
    model.optimizer.zero_grad()

    return loss.item(), metric.item()

# 测试train_step效果
features,labels = next(iter(dl))
train_step(model,features,labels)

def train_model(model,epochs):
    for epoch in range(1,epochs+1):
        loss_list,metric_list = [],[]
        for features, labels in dl:
            lossi,metrici = train_step(model,features,labels)
            loss_list.append(lossi)
            metric_list.append(metrici)
        loss = np.mean(loss_list)
        metric = np.mean(metric_list)

        if epoch%100==0:
            printbar()
            print("epoch =",epoch,"loss = ",loss,"metric = ",metric)

train_model(model,epochs = 300)
