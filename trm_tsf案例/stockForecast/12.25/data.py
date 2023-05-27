import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

#最后的效果 x---[batch,seq,feature_size]   y---[batch,seq]

data_path=r'.\600243.csv'
elements=['收盘价','最高价','最低价','开盘价','前收盘']
element='开盘价'

def single_data():#以收盘价为y，且x归一化
    data_all = pd.read_csv(data_path, encoding='gbk')
    data_ha = []
    length = len(data_all)
    for index, element in enumerate(elements):
        data_element = data_all[element].values.astype(np.float64)
        data_element = data_element.reshape(length, 1)
        data_ha.append(data_element)
    X_hat = np.concatenate(data_ha, axis=1)       #[none,feture_size]
    print(X_hat.shape)
    # X_hat=data_all[element].values.astype(np.float64)
    max1 = np.max(X_hat,axis=0)
    print(max1)
    # X_hat=X_hat.reshape(-1,1)
    temp = np.zeros(shape=(X_hat.shape[0],X_hat.shape[1]))
    print(temp.shape)
    a = X_hat.shape[0]
    for i in range(a):
        temp[i, :] = X_hat[a - 1 - i, :]
    y = temp[5:,3]
    # y = temp[5:]
    if len(y.shape)<2:
        y = np.expand_dims(y,1)
    X=temp/max1
    X = X[0:-5, :]
    return X,y     #[none,feature_size]  [none,feature_size]默认out_size为1

def data_load(seq_len):
    x,y=single_data()
    len=x.shape[0]
    data_last_index=len-seq_len
    X=[]
    Y=[]
    for i in range(0,data_last_index,5):
        data_x=np.expand_dims(x[i:i+seq_len],0)   #[1,seq,feature_size]
        data_y=np.expand_dims(y[i:i+seq_len],0)   #[1,seq,out_size]
        # data_y=np.expand_dims(y[,0)   #[1,seq,out_size]
        X.append(data_x)
        Y.append(data_y)
    data_x= np.concatenate(X, axis=0)
    data_y=np.concatenate(Y, axis=0)
    data=torch.from_numpy(data_x).type(torch.float32)
    label=torch.from_numpy(data_y).type(torch.float32)
    return data,label    #[num_data,seq,feature_size]  [num_data,seq] 默认out_size为1

def dataset(seq_len,batch):
    X,Y=data_load(seq_len)
    feature_size=X.shape[-1]
    out_size=Y.shape[-1]
    dataset_train=TensorDataset(X,Y)
    dataloader=DataLoader(dataset_train,batch_size=batch,shuffle=False)
    return dataloader,feature_size,out_size