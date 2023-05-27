from transformerhah import Transformer
import torch.nn as nn
import torch.optim as optim
from data import single_data
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import copy

png_save_path=r'.\png\5'
if not os.path.isdir(png_save_path):
    os.mkdir(png_save_path)

path_train=os.path.join(png_save_path,'weight.pth')

loader=single_data()

model = Transformer()
criterion = nn.MSELoss()     #忽略 占位符 索引为0.
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

best_loss=100000
best_epoch=0
for epoch in range(50):
    epoch_loss=0
    y_pre=[]
    y_true=[]
    for X,y in loader:  # enc_inputs : [batch_size, src_len,1](64*5)
        enc_inputs=X.unsqueeze(0)   #(1*64*5)
        # enc_inputs=enc_inputs.squeeze(2)
        # dec_inputs : [batch_size, ]
        # dec_outputs: [batch_size, 1]
        outputs, enc_self_attns = model(enc_inputs)
        # print(outputs.shape)
        outputs=outputs.squeeze(1)
        outputs=outputs.unsqueeze(0)
        y=y.unsqueeze(0)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        loss = criterion(outputs, y.view(1,-1))
        loss_num=loss.item()
        epoch_loss+=loss_num
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        y_pre.append(outputs.detach().numpy())
        y_true.append(y.detach().numpy())

    if epoch_loss<best_loss:
        best_loss=epoch_loss
        best_epoch=epoch
        best_model_wts=copy.deepcopy(model.state_dict())
        torch.save(best_model_wts,path_train)

    pre = np.concatenate(y_pre, axis=1).squeeze(0)  # no norm label
    true = np.concatenate(y_true, axis=1).squeeze(2)  # no norm label
    true=true.squeeze(0)
    if True:
        plt.plot(true, color="blue", alpha=0.5)
        plt.plot(pre, color="red", alpha=0.5)
        plt.plot(pre - true, color="green", alpha=0.8)
        plt.grid(True, which='both')
        plt.axhline(y=0, color='k')
        # pyplot.savefig(os.path.join(png_save_path, 'pre.png'))
        plt.savefig(os.path.join(png_save_path, '%d.png'%epoch))
        plt.close()
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss))
print('best_loss::|',best_loss,'---best_epoch::|',best_epoch)
train_over_path=os.path.join(png_save_path,'loss(%d)---epoch(%d).pth'%(best_loss,best_epoch))
os.rename(path_train,train_over_path)
print('*******************over****************************')