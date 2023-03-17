import os
import datetime
import torch
from torch import nn
import torch.nn.functional as F

'''
Pytorch和神经网络相关的功能组件大多都封装在 torch.nn模块下
这些功能组件的绝大部分既有函数形式实现，也有类形式实现
'''

# nn.functional(一般引入后改名为F)有各种功能组件的函数实现
'''
(激活函数)
* F.relu
* F.sigmoid
* F.tanh
* F.softmax

(模型层)
* F.linear
* F.conv2d
* F.max_pool2d
* F.dropout2d
* F.embedding

(损失函数)
* F.binary_cross_entropy
* F.mse_loss
* F.cross_entropy
'''

# 为了便于对参数进行管理，一般通过继承 nn.Module 转换成为类的实现形式，并直接封装在 nn 模块下
# nn.Module除了可以管理其引用的各种参数，还可以管理其引用的子模块
'''
(激活函数)
* nn.ReLU
* nn.Sigmoid
* nn.Tanh
* nn.Softmax

(模型层)
* nn.Linear
* nn.Conv2d
* nn.MaxPool2d
* nn.Dropout2d
* nn.Embedding

(损失函数)
* nn.BCELoss
* nn.MSELoss
* nn.CrossEntropyLoss
'''

# 使用nn.Module来管理参数
# nn.Parameter 具有 requires_grad = True 属性
w = nn.Parameter(torch.randn(2,2))
print(w)
print(w.requires_grad)

# nn.ParameterList 可以将多个nn.Parameter组成一个列表
params_list = nn.ParameterList([nn.Parameter(torch.rand(8,i)) for i in range(1,3)])
print(params_list)
print(params_list[0].requires_grad)

# nn.ParameterDict 可以将多个nn.Parameter组成一个字典

params_dict = nn.ParameterDict({"a":nn.Parameter(torch.rand(2,2)),
                               "b":nn.Parameter(torch.zeros(2))})
print(params_dict)
print(params_dict["a"].requires_grad)

# 1. 可以用Module将它们管理起来
# module.parameters()返回一个生成器，包括其结构下的所有parameters
module = nn.Module()
module.w = w
module.params_list = params_list
module.params_dict = params_dict

num_params = 0
for param in module.parameters():
    print(param,"\n")
    num_param = num_param + 1
print("number of Parameters =",num_param)

#实践当中，一般通过继承nn.Module来构建模块类，并将所有含有需要学习的参数的部分放在构造函数中。
#以下范例为Pytorch中nn.Linear的源码的简化版本
#可以看到它将需要学习的参数放在了__init__构造函数中，并在forward中调用F.linear函数来实现计算逻辑。
class Linear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        return F.linear(inputs,self.weight, self.bias)

# 2. 使用nn.Module来管理子模块
'''
很少直接使用 nn.Parameter来定义参数构建模型, 而是通过一些拼装一些常用的模型层来构造模型。
这些模型层也是继承自nn.Module的对象,本身也包括参数，属于我们要定义的模块的子模块。
nn.Module提供了一些方法可以管理这些子模块:

children() 方法: 返回生成器，包括模块下的所有子模块
named_children()方法：返回一个生成器，包括模块下的所有子模块，以及它们的名字
modules()方法：返回一个生成器，包括模块下的所有各个层级的模块，包括模块本身
named_modules()方法：返回一个生成器，包括模块下的所有各个层级的模块以及它们的名字，包括模块本身


其中chidren()方法和named_children()方法较多使用。
modules()方法和named_modules()方法较少使用,其功能可以通过多个named_children()的嵌套使用实现
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=3, padding_idx=1)
        self.conv = nn.Sequential()
        self.conv.add_module('conv_1', nn.Conv1d(in_channels=3, out_channels=16, kernel_size=5))
        self.conv.add_module('pool_1', nn.MaxPool1d(kernel_size=2))
        self.conv.add_module('relu_1', nn.ReLU())
        self.conv.add_module('conv_2', nn.Conv1d(in_channels=16, out_channels=128, kernel_size=2))
        self.conv.add_module('pool_2', nn.MaxPool1d(kernel_size=2))
        self.conv.add_module('relu_2', nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module('flatten', nn.Flatten())
        self.dense.add_module('linera', nn.Linear(6144, 1))
        self.dense.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()
i = 0
for child in net.children():
    i += 1
    print(child, '\n')
print("child number",i) # child number 3[Embedding, Sequential, Sequential]

i = 0
for name,child in net.named_children():
    i+=1
    print(name,":",child,"\n")
print("child number",i) # child number 3 [embedding : Embedding, conv : Sequential, dense : Sequential]

i = 0
for module in net.modules():
    i+=1
    print(module)
print("module number:",i)

# 通过named_children方法找到embedding层，并将其参数设置为不可训练(相当于冻结embedding层)
children_dict = {name:module for name, module in net.named_children()}
embedding = children_dict['embedding']
embedding.requires_grad_(False)

'''
print(children_dict)
{'embedding': Embedding(10000, 3, padding_idx=1), 
'conv': Sequential(
  (conv_1): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (pool_1): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_1): ReLU()
  (conv_2): Conv1d(16, 128, kernel_size=(2,), stride=(1,))
  (pool_2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (relu_2): ReLU()
), 
'dense': Sequential(
  (flatten): Flatten()
  (linear): Linear(in_features=6144, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)}
'''

for param in embedding.parameters():
    print(param.requires_grad) # False
    print(param.numel()) # 30000




'''
from torchkeras import summary
summary(net, input_shape=(200,), input_dtype=torch.LongTensor)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Embedding-1               [-1, 200, 3]          30,000
            Conv1d-2              [-1, 16, 196]             256
         MaxPool1d-3               [-1, 16, 98]               0
              ReLU-4               [-1, 16, 98]               0
            Conv1d-5              [-1, 128, 97]           4,224
         MaxPool1d-6              [-1, 128, 48]               0
              ReLU-7              [-1, 128, 48]               0
           Flatten-8                 [-1, 6144]               0
            Linear-9                    [-1, 1]           6,145
          Sigmoid-10                    [-1, 1]               0
================================================================
Total params: 40,625
Trainable params: 10,625
Non-trainable params: 30,000
----------------------------------------------------------------
Input size (MB): 0.000763
Forward/backward pass size (MB): 0.287796
Params size (MB): 0.154972
Estimated Total Size (MB): 0.443531
----------------------------------------------------------------
'''

