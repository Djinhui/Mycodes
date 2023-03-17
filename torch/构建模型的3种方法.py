'''
可以使用以下3种方式构建模型:

1 继承nn.Module基类构建自定义模型。
2 使用nn.Sequential按层顺序构建模型。
3 继承nn.Module基类构建模型并辅助应用模型容器进行封装(nn.Sequential,nn.ModuleList,nn.ModuleDict)
'''
import torch 
from torch import nn
# from torchkeras import summary

# 一，继承nn.Module基类构建自定义模型
'''
模型中的用到的层一般在__init__函数中定义,然后在forward方法中定义模型的正向传播逻辑
'''
class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #  [-1, 64, 5, 5]
        self.dropout = nn.Dropout2d(p=0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1)) #  [-1, 64, 1, 1]
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        y = self.sigmoid(x)
        return y

net = Net()
print(net)

# 二，使用nn.Sequential按层顺序构建模型
# 使用nn.Sequential按层顺序构建模型无需定义forward方法。仅仅适合于简单的模型
# 1. 使用add_module()
net = nn.Sequential()
net.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
net.add_module("pool1",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("conv2",nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5))
net.add_module("pool2",nn.MaxPool2d(kernel_size = 2,stride = 2))
net.add_module("dropout",nn.Dropout2d(p = 0.1))
net.add_module("adaptive_pool",nn.AdaptiveMaxPool2d((1,1)))
net.add_module("flatten",nn.Flatten())
net.add_module("linear1",nn.Linear(64,32))
net.add_module("relu",nn.ReLU())
net.add_module("linear2",nn.Linear(32,1))
net.add_module("sigmoid",nn.Sigmoid())

print(net)

# 2. 利用变长参数, 这种方式构建时不能给每个层指定名称
net = nn.Sequential(
    nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
    nn.MaxPool2d(kernel_size = 2,stride = 2),
    nn.Dropout2d(p = 0.1),
    nn.AdaptiveMaxPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(64,32),
    nn.ReLU(),
    nn.Linear(32,1),
    nn.Sigmoid()
)

print(net)

# 3. 利用OrderedDict
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3))
]))


# 三，继承nn.Module基类构建模型并辅助应用模型容器进行封装

# 模型容器(nn.Sequential,nn.ModuleList,nn.ModuleDict)
'''
在下面的范例中我们每次仅仅使用一种模型容器，但实际上这些模型容器的使用是非常灵活的，可以在一个模型中任意组合任意嵌套使用
'''
# 1，nn.Sequential作为模型容器
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1))
        )
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.conv(x)
        y = self.dense(x)
        return y 

net = Net()
print(net)

# 2，nn.ModuleList作为模型容器
# 下面中的ModuleList不能用Python中的列表代替
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
            nn.MaxPool2d(kernel_size = 2,stride = 2),
            nn.Dropout2d(p = 0.1),
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.Sigmoid()]
        )

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
net = Net()
print(net)

# 3，nn.ModuleDict作为模型容器
# 注意下面中的ModuleDict不能用Python中的字典代替
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layers_dict = nn.ModuleDict(
            {"conv1":nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3),
               "pool": nn.MaxPool2d(kernel_size = 2,stride = 2),
               "conv2":nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5),
               "dropout": nn.Dropout2d(p = 0.1),
               "adaptive":nn.AdaptiveMaxPool2d((1,1)),
               "flatten": nn.Flatten(),
               "linear1": nn.Linear(64,32),
               "relu":nn.ReLU(),
               "linear2": nn.Linear(32,1),
               "sigmoid": nn.Sigmoid()
              })

    def forward(self,x):
        layers = ["conv1","pool","conv2","pool","dropout","adaptive",
                  "flatten","linear1","relu","linear2","sigmoid"]
        for layer in layers:
            x = self.layers_dict[layer](x)
        return x
net = Net()
print(net)