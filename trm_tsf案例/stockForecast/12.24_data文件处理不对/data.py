import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

data_path = './600243.csv'
elements = ['收盘价','最高价','最低价','开盘价','前收盘']
# elements = ['开盘价]

def single_data(): # 以收盘价为y

    '''
    data_all = pd.read_csv(data_path, encoding='gbk')
    data_all['日期'] = pd.to_datetime(data_all['日期'])
    data_all.sort_values(by=['日期'],ascending=True)
    X_hat = data_all[elements].values.astype(np.float64)
    '''
    data_all = pd.read_csv(data_path, encoding='gbk')
    data_ha = []
    length = len(data_all)
    for index, element in enumerate(elements):
        data_element = data_all[element].values.astype(np.float64)
        data_element = data_element.reshape(length, 1)
        data_ha.append(data_element)

    X_hat = np.concatenate(data_ha, axis=1)
    X_CONVERT = torch.from_numpy(X_hat)
    X = torch.zeros_like(X_CONVERT)
    a = len(X_CONVERT)
    for i in range(a):
        X[i, :] = X_CONVERT[a-1-i, :] # 原始数据是降序的
    y = X[5:, 3].type(torch.float32)
    y = y.reshape(y.shape[0], 1)
    X = X[0:-5, :].type(torch.float32)
    # X-=torch.min(X,dim=0)
    # X/=torch.max(X,dim=0)
    # X -= torch.mean(X, dim=0)
    # X /= torch.std(X, dim=0)
    dataset=TensorDataset(X,y)
    data_loader=DataLoader(dataset,batch_size=64,shuffle=False)
    return data_loader   #torch.Size([64, 5]) [64,1])