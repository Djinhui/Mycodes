'''
比如我们利用前24小时的[负荷、温度、湿度、压强]预测接下来12个时刻的[负荷、温度、湿度、压强]。
实际上，我们可以将多个变量的输出分解开来，看成多个任务，也就是多任务学习，其中每一个任务都是前面提到的多变量输入单变量输出
'''
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import chain
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, n_outputs=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.n_outputs = n_outputs
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.fcs = [nn.Linear(self.hidden_size, self.output_size).to(device) for i in range(self.n_outputs)]
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        # print(input_seq.shape)
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        preds = []
        pred1, pred2, pred3 = self.fc1(output), self.fc2(output), self.fc3(output)
        pred1, pred2, pred3 = pred1[:, -1, :], pred2[:, -1, :], pred3[:, -1, :]

        # pred = torch.cat([pred1, pred2], dim=0)
        pred = torch.stack([pred1, pred2, pred3], dim=0)
        # print(pred.shape)

        return pred 

'''
pred_step_size = output_size
label shape(batch_size, n_outputs, pred_step_size)
pred shape((n_outputs, batch_size, pred_step_size))

total_loss = 0
for k in range(args.n_outputs):
    total_loss = total_loss + loss_function(preds[k, :, :], labels[:, k, :])
total_loss /= preds.shape[0]
'''