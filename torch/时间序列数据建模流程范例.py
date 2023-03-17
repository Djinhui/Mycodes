import os
import datetime
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchkeras 

#打印时间
def printbar():
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n"+"=========="*8 + "%s"%nowtime)

#mac系统上pytorch和matplotlib在jupyter中同时跑需要更改环境变量
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" 


# 1. 准备数据
'''
通过继承torch.utils.data.Dataset实现自定义时间序列数据集
'''
df = pd.read_csv("../data/covid-19.csv",sep = "\t")
df.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
plt.xticks(rotation=60)

dfdata = df.set_index("date")
dfdiff = dfdata.diff(periods=1).dropna()
dfdiff = dfdiff.reset_index("date")

dfdiff.plot(x = "date",y = ["confirmed_num","cured_num","dead_num"],figsize=(10,6))
plt.xticks(rotation=60)
dfdiff = dfdiff.drop("date",axis = 1).astype("float32")

dfdiff.shape # (None, 3)

# 用某日前8天窗口数据作为输入预测该日数据
WINDOW_SIZE = 8
class Covid19Dataset(Dataset):
    def __len__(self):
        return len(dfdiff) - WINDOW_SIZE

    def __getitem__(self, index):
        x = dfdiff.loc[index:index+WINDOW_SIZE-1, :]
        featrue = torch.tensor(x.values)
        y = dfdiff.loc[index+WINDOW_SIZE, :]
        label = torch.tensor(y.values)

        return (featrue, label)


ds_train = Covid19Dataset()
#数据较小，可以将全部训练数据放入到一个batch中，提升性能
dl_train = DataLoader(ds_train,batch_size = 38)

# 2. 定义模型
torch.random.seed()

class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,x, x_input):
        x_out = torch.max((1+x)*x_input[:,-1,:], torch.tensor(0.0))
        return x_out

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=3, num_layers=5, batch_first = True)
        self.linear = nn.Linear(3,3)
        self.block = Block()

    def forward(self, x_input):
        x = self.lstm(x_input)[0][:,-1,:] # [-1, 8, 3] --> [-1, 3]
        x = self.linear(x)
        y = self.block(x, x_input)
        return y

net = Net()
model = torchkeras.Model(net)
print(model)

model.summary(input_shape=(8,3),input_dtype = torch.FloatTensor)

# 3. torchkeras:我们仿照Keras定义了一个高阶的模型接口Model
def mspe(y_pred,y_true):
    err_percent = (y_true - y_pred)**2/(torch.max(y_true**2,torch.tensor(1e-7)))
    return torch.mean(err_percent)

model.compile(loss_func = mspe,optimizer = torch.optim.Adagrad(model.parameters(),lr = 0.1))
dfhistory = model.fit(100,dl_train,log_step_freq=10)

# 4. 评估模型
import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.title('Training '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric])
    plt.show()

plot_metric(dfhistory,"loss")

# 5. 使用模型
# 使用dfresult记录现有数据以及此后预测的疫情数据
dfresult = dfdiff[["confirmed_num","cured_num","dead_num"]].copy()
dfresult.tail()

#预测此后200天的新增走势,将其结果添加到dfresult中
for i in range(200):
    arr_input = torch.unsqueeze(torch.from_numpy(dfresult.values[-38:,:]),axis=0)
    arr_predict = model.forward(arr_input)

    dfpredict = pd.DataFrame(torch.floor(arr_predict).data.numpy(),
                columns = dfresult.columns)
    dfresult = dfresult.append(dfpredict,ignore_index=True)

dfresult.query("confirmed_num==0").head()

# 第50天开始新增确诊降为0，第45天对应3月10日，也就是5天后，即预计3月15日新增确诊降为0
# 注：该预测偏乐观

# 6. 保存模型
# 保存模型参数

torch.save(model.net.state_dict(), "../data/model_parameter.pkl")

net_clone = Net()
net_clone.load_state_dict(torch.load("../data/model_parameter.pkl"))
model_clone = torchkeras.Model(net_clone)
model_clone.compile(loss_func = mspe)

# 评估模型
model_clone.evaluate(dl_train)