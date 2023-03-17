'''
Pytorch的计算图由节点和边组成,节点表示张量或者Function,边表示张量和Function之间的依赖关系。

Pytorch中的计算图是动态图。这里的动态主要有两重含义。
第一层含义是：计算图的正向传播是立即执行的。无需等待完整的计算图创建完毕，每条语句都会在计算图中动态添加节点和边，
并立即执行正向传播得到计算结果。

第二层含义是:计算图在反向传播后立即销毁。下次调用需要重新构建计算图。
如果在程序中使用了backward方法执行了反向传播,或者利用torch.autograd.grad方法计算了梯度,
那么创建的计算图会被立即销毁,释放存储空间，下次调用需要重新创建。
'''
import numpy as np
import torch

# 1. 计算图的正向传播是立即执行的
w = torch.tensor([[3.0,1.0]], requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.randn(10,2)
Y = torch.randn(10,1)
Y_hat = X@w.t() + b  # Y_hat定义后其正向传播被立即执行，与其后面的loss创建语句无关
loss = torch.mean(torch.pow(Y-Y_hat, 2))
print(loss.data)
print(Y_hat.data)


# 2. 计算图在反向传播后立即销毁
#计算图在反向传播后立即销毁，如果需要保留计算图, 需要设置retain_graph = True
loss.backward() # loss.backward(retain_graph = True) 
# loss.backward() # 如果再次执行反向传播将报错


# 3. 计算图中的Function
'''
计算图中的另外一种节点是Function, 实际上就是 Pytorch中各种对张量操作的函数
Function和Python中的函数有一个较大的区别,那就是它同时包括正向计算逻辑和反向传播的逻辑
'''

class MyReLU(torch.autograd.Function):

    #正向传播逻辑，可以用ctx存储一些值，供反向传播使用。
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    #反向传播逻辑
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

import torch 
w = torch.tensor([[3.0,1.0]],requires_grad=True)
b = torch.tensor([[3.0]],requires_grad=True)
X = torch.tensor([[-1.0,-1.0],[1.0,1.0]])
Y = torch.tensor([[2.0,3.0]])

relu = MyReLU.apply # relu现在也可以具有正向传播和反向传播功能
Y_hat = relu(X@w.t() + b)
loss = torch.mean(torch.pow(Y_hat-Y,2))

loss.backward()

print(w.grad)
print(b.grad)

# Y_hat的梯度函数即是我们自己所定义的 MyReLU.backward
print(Y_hat.grad_fn)


import torch 

x = torch.tensor(3.0,requires_grad=True)
y1 = x + 1
y2 = 2*x
loss = (y1-y2)**2

loss.backward()
print("loss.grad:", loss.grad)
print("y1.grad:", y1.grad)
print("y2.grad:", y2.grad)
print(x.grad)

'''
loss.grad: None
y1.grad: None
y2.grad: None
tensor(4.)
'''

print(x.is_leaf)
print(y1.is_leaf)
print(y2.is_leaf)
print(loss.is_leaf)
'''
True
False
False
False
'''

# 5. 计算图在TensorBoard中的可视化
from torch import nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.w = nn.Parameter(torch.randn(2,1))
        self.b = nn.Parameter(torch.zeros(1,1))

    def forward(self, x):
        y = x@self.w + self.b
        return y

net = Net()

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('../data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(10,2))
writer.close()

# %load_ext tensorboard
# #%tensorboard --logdir ../data/tensorboard

# from tensorboard import notebook
# notebook.list() 

# #在tensorboard中查看模型
# notebook.start("--logdir ../data/tensorboard")