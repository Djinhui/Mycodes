import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

df = pd.DataFrame() # regrad this df as data

def generate_sequences(df:pd.DataFrame, tw:int, pw:int, target_columns, drop_targets=False):
    '''
    df: Pandas DataFrame of the univariate time-series
    tw: Training Window - Integer defining how many steps to look back
    pw: Prediction Window - Integer defining how many steps forward to predict


    returns: dictionary of sequences and targets for all sequences
    '''

    data = dict()
    l = len(df)
    for i in range(l-tw):
        # if drop_targets:
        #     df.drop(target_columns, axis=1, inplace=True)
        sequence = df[i:i+tw].values
        target = df[i+tw:i+tw+pw][target_columns].values

        data[i] = {'sequence':sequence, 'target':target}
    return data


class SequenceDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __getitem__(self, index):
        sample = self.data[index]
        return torch.Tensor(sample['sequence']), torch.Tensor(sample['target'])

    def __len__(self):
        return len(self.data)


nout = 1 # Prediction Window
sequence_len = 180 # Training Window
BATCH_SIZE = 16 # Training batch size
split = 0.8 # Train/Test Split ratio

sequences = generate_sequences(df, sequence_len, nout,'dcoilwtico')
dataset = SequenceDataset(sequences)

train_len = int(len(dataset)*split)
lens = [train_len, len(dataset)-train_len]
train_ds, test_ds = random_split(dataset, lens)
trainloader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
testloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1,n_deep_layers=10, use_cuda=False, dropout=0.2):
        '''
        n_features: number of input features (1 for univariate forecasting)
        n_hidden: number of neurons in each hidden layer
        n_outputs: number of outputs to predict for each training example
        n_deep_layers: number of hidden dense layers after the lstm layer
        sequence_len: number of steps to look back at for prediction
        dropout: float (0 < dropout < 1) dropout ratio between dense layers
        '''
        super(LSTMForecaster, self).__init__()
        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden
        self.use_cuda = use_cuda
        self.lstm = nn.LSTM(n_features, n_hidden, num_layers=n_lstm_layers, batch_first=True)

        self.fc1 = nn.Linear(n_hidden*sequence_len, n_hidden)
        self.dropout = nn.Dropout(p=dropout)

        dnn_layers = []
        for i in range(n_deep_layers):
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_outputs))
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(n_hidden, n_hidden))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))

        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        if self.use_cuda:
            hidden_state = hidden_state.to('cuda:0')
            cell_state = cell_state.to('cuda:0')

        self.hidden = (hidden_state, cell_state)

        x, h = self.lstm(x, self.hidden)
        x = self.dropout(x.contiguous().view(x.shape[0], -1))
        x = self.fc1(x)
        return self.dnn(x)


nhid = 50 # Number of nodes in the hidden layer
n_dnn_layers = 5 # Number of hidden fully connected layers
# Number of features (since this is a univariate timeseries we'll set
# this to 1 -- multivariate analysis is coming in the future)
ninp = 1

# Device selection (CPU | GPU)
USE_CUDA = torch.cuda.is_available()
device = 'cuda' if USE_CUDA else 'cpu'

model = LSTMForecaster(ninp, nhid, nout, sequence_len, n_deep_layers=n_dnn_layers,use_cuda=USE_CUDA).to(device)

lr = 4e-4
n_epochs = 20
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

t_losses, v_losses = [], []
for epoch in range(n_epochs):
    train_loss, valid_loss = 0.0, 0.0
    model.train()
    for x, y in trainloader:
        optimizer.zero_grad()
        x = x.to(device)
        y = y.squeeze().to(device)
        preds = model(x).squeeze()
        loss = criterion(preds, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_loss = train_loss / len(trainloader)
    t_losses.append(epoch_loss)

    model.eval()
    for x, y in testloader:
        with torch.no_grad():
            x, y = x.to(device), y.squeeze().to(device)
            preds = model(x).squeeze()
            error = criterion(preds, y)
        valid_loss += error.item()

    valid_loss = valid_loss / len(testloader)
    v_losses.append(valid_loss)

    print(f'{epoch} - train: {epoch_loss}, valid: {valid_loss}')


# plt.plot(t_losses)
# plt.plot(v_losses)
def make_pred(model, unshuffled_dl):
    model.eval()
    preds, acts = [], []
    for x, y in unshuffled_dl:
        with torch.no_grad():
            p = model(x)
            preds.append(p)
            acts.append(y.squeeze())
    preds = torch.cat(preds).numpy()
    acts = torch.cat(acts).numpy()
    return preds.squeeze(), acts


'''
预测
1. 从历史(训练窗口长度)中获取最新的有效序列。
2. 将最新的序列输入模型并预测下一个值。
3. 将预测值附加到历史记录上。
4. 迭代重复步骤1。
'''
def one_step_forecast(model, history):
    model.cpu()
    model.eval()
    with torch.no_grad():
        pre = torch.Tensor(history).unsqueeze(0)
        pred = model(pre)
    return pred.detach().numpy().reshape(-1)

def n_step_forecast(data: pd.DataFrame, target: str, tw: int, n: int, forecast_from: int=None, plot=False):
      '''
      n: integer defining how many steps to forecast
      forecast_from: integer defining which index to forecast from. None if
      you want to forecast from the end.
      plot: True if you want to output a plot of the forecast, False if not.
      '''
      history = data[target].copy().to_frame()


      # Create initial sequence input based on where in the series to forecast from.
      if forecast_from:
        pre = list(history[forecast_from - tw : forecast_from][target].values)
      else:
        pre = list(history[target])[-tw:]


      # Call one_step_forecast n times and append prediction to history
      for i, step in enumerate(range(n)):
        pre_ = np.array(pre[-tw:]).reshape(-1, 1)
        forecast = one_step_forecast(pre_).squeeze()
        pre.append(forecast)


      # The rest of this is just to add the forecast to the correct time of 
      # the history series
      res = history.copy()
      ls = [np.nan for i in range(len(history))]


      # Note: I have not handled the edge case where the start index + n is 
      # before the end of the dataset and crosses past it.
      if forecast_from:
        ls[forecast_from : forecast_from + n] = list(np.array(pre[-n:]))
        res['forecast'] = ls
        res.columns = ['actual', 'forecast']
      else:
        fc = ls + list(np.array(pre[-n:]))
        ls = ls + [np.nan for i in range(len(pre[-n:]))]
        ls[:len(history)] = history[target].values
        res = pd.DataFrame([ls, fc], index=['actual', 'forecast']).T
      return res