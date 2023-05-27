# https://blog.csdn.net/Cyril_KI/article/details/122569775

# 单步预测


import pandas as pd
import numpy as np
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import chain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(filename):
    df = pd.read_csv(filename)
    #  do some preprocess
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]
        
    def __len__(self):
        return len(self.data)
    
# 原始是单变量预测:根据前24个时刻的负荷下一时刻的负荷,这里指定feature_names
def nn_seq_us(batch_size, feature_names=[], target_names=[]):
    dataset = load_data('xxx')
    train = dataset[:int(len(dataset) * 0.6)]
    val = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.8)]
    test = dataset[int(len(dataset) * 0.8):len(dataset)]
    
    def process(data, batch_size, shuffle):
        load = data[feature_names+target_names]
        # scaler.fit_transform(load)
        seq = []
        for i in range(len(data)-24):
            train_seq = []
            train_label = []
            for j in range(i, i+24):
                x = [load.loc[i, feature_names].values().tolist()]
                train_seq.append(x)

            train_label.append(load.loc[i+24, target_names].values.tolist())
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(-1)
            seq.append((train_seq, train_label))

        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True)
        return seq
    
    Dtr = process(train, batch_size, True)
    Dval = process(val, batch_size, True)
    Dte = process(test, batch_size, False)

    return Dtr, Dval, Dte

'''
注意：
nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.5)
如果num_layers=2, hidden_size=64,那么两层LSTM的hidden_size都为64,并且最后一层也就是第二层结束后不会执行dropout策略。
如果我们需要让两层LSTM的hidden_size不一样,并且每一层后都执行dropout,就可以采用LSTMCell(每一步的操作)来实现多层的LSTM。

class LSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm0 = nn.LSTMCell(args.input_size, hidden_size=128)
        self.lstm1 = nn.LSTMCell(input_size=128, hidden_size=32)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(32, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # batch_size, hidden_size
        h_l0 = torch.zeros(batch_size, 128).to(device)
        c_l0 = torch.zeros(batch_size, 128).to(device)
        h_l1 = torch.zeros(batch_size, 32).to(device)
        c_l1 = torch.zeros(batch_size, 32).to(device)
        output = []
        for t in range(seq_len): # 沿时间轴循环
            h_l0, c_l0 = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))
            h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)
            h_l1, c_l1 = self.lstm1(h_l0, (h_l1, c_l1))
            h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
            output.append(h_l1)

        pred = self.linear(output[-1])

        return pred

'''
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)  # pred()
        pred = pred[:, -1, :]
        
        return pred

    
class BiLSTM2(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        output = torch.mean(output, dim=2)
        pred = self.linear(output)
        # print('pred=', pred.shape)
        pred = pred[:, -1, :]
        
        return pred




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size  # len(feature_names)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size # len(target_names)
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions*self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions*self.num_layers, self.batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        # output(batch_size, seq_len, num_directions * hidden_size)
        pred = self.linear(output) # (batch_size, seq_len, output_size)
        pred = pred[:, -1, :]
        return pred
    
    
def train(args, Dtr, Val, modelpath):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM2(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    loss_fn = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    min_epochs = 10
    best_model = None
    min_val_loss = 5
    model.train()
    for epoch in range(args.epochs):
        train_loss = []
        for (seq, label) in Dtr:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_fn(y_pred, label)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        def get_val_loss(args, model, Val):
            model.eval()
            val_losses = []
            for (seq, label) in Val:
                seq = seq.to(device)
                label = label.to(device)
                y_pred = model(seq)
                loss = loss_fn(y_pred, label)
                val_loss.append(loss.item())
            return np.mean(val_losses)
            

        # if epoch % 5 == 0:
        val_loss = get_val_loss(args, model,Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)

        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'model':best_model.state_dict}
    torch.save(state, modelpath)



def test(args, Dte, modelpath):
    pred = []
    y = []

    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.bidirectional:
        model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)

    model.load_state_dict(torch.load(modelpath)['model'])
    model.eval()
    for (seq, target) in Dte:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(target.data.tolist()))
            pred.extend(y_pred)

    
    y, pred = np.array(y), np.array(pred)
    return y, pred



    
        
