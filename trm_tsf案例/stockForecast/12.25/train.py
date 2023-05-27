from model import TransAm
import torch.nn as nn
import torch
import torch.optim as optim
from matplotlib import pyplot
import numpy as np
import os
from data import dataset

png_save_path=r'.\png\5'

seq_len=64
batch=8
loader,feature_size,out_size=dataset(seq_len,batch)

model = TransAm(feature_size,out_size)
criterion = nn.MSELoss()     #忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)


def train(epochs,isplot):
    for epoch in range(epochs):
        epoch_loss = 0
        y_pre = []
        y_true = []
        for X, y in loader:  #X--[batch,seq,feature_size]  y--[batch,seq]
            enc_inputs = X.permute([1,0,2])  #[seq,batch,feature_size]
            y=y.permute([1,0,2])
            key_padding_mask = torch.ones(enc_inputs.shape[1], enc_inputs.shape[0])  # [batch,seq]
            optimizer.zero_grad()
            output = model(enc_inputs, key_padding_mask)
            #output--[seq,batch,out_size] enc_self_attns--[seq,seq]
            output=output[-5:]
            y=y[-5:]
            loss=criterion(output,y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss+=loss.item()
            pres=output.detach().numpy()  #[seq,batch,out_size]
            pres=pres.transpose(0,1,2).reshape(-1,out_size)   #[none,out_size] # torch的transpose只接受两个维度(dim0, dim1)
            tru=y.detach().numpy()
            tru=tru.transpose(0,1,2).reshape(-1,out_size) #  torch的transpose只接受两个维度(dim0, dim1)
            y_pre.append(pres)
            y_true.append(tru)
        pre=np.concatenate(y_pre,axis=0)
        true=np.concatenate(y_true,axis=0)
        if isplot:
            pyplot.plot(true, color="blue", alpha=0.5)
            pyplot.plot(pre, color="red", alpha=0.5)
            pyplot.plot(pre - true, color="green", alpha=0.8)
            pyplot.grid(True, which='both')
            pyplot.axhline(y=0, color='k')
            # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
            pyplot.savefig(os.path.join(png_save_path, '%d.png' % epoch))
            pyplot.close()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss))

train(30,True)