# Pytorch中使用GPU只要将模型和数据移动到GPU上
'''
单GPU
# 定义模型
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device) # 移动模型到cuda

# 训练模型

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者  labels = labels.cuda() if torch.cuda.is_available() else labels

多GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model) # 包装为并行风格模型

features = features.to(device) # 移动数据到cuda
labels = labels.to(device) # 或者 labels = labels.cuda() if torch.cuda.is_available() else labels
'''
import torch 
from torch import nn 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1，查看gpu信息
if_cuda = torch.cuda.is_available()
print("if_cuda=",if_cuda)

gpu_count = torch.cuda.device_count()
print("gpu_count=",gpu_count)

# 2，将张量在gpu和cpu间移动
tensor = torch.rand((100,100))
tensor_gpu = tensor.to('cuda:0') 
tensor_gpu = tensor.cuda()

tensor_cpu = tensor_gpu.to('cpu') 
tensor_cpu = tensor_gpu.cpu()

# 3，将模型中的全部张量移动到gpu上
net = nn.Linear(2,1)
print(next(net.parameters()).is_cuda)
net.to("cuda:0") # 将模型中的全部参数张量依次到GPU上，注意，无需重新赋值为 net = net.to("cuda:0")
print(next(net.parameters()).is_cuda)
print(next(net.parameters()).device)

# 4，创建支持多个gpu数据并行的模型
linear = nn.Linear(2,1)
print(next(linear.parameters()).device)

model = nn.DataParaller(linear)
print(model.device_ids)
print(next(model.module.parameters()).device) 

#注意保存参数时要指定保存model.module的参数
torch.save(model.module.state_dict(), "../data/model_parameter.pkl") 

linear = nn.Linear(2,1)
linear.load_state_dict(torch.load("../data/model_parameter.pkl")) 

# 5，清空cuda缓存

# 该方法在cuda超内存时十分有用
torch.cuda.empty_cache()

# 一，矩阵乘法范例
# 使用cpu
import time

a = torch.rand((10000,200))
b = torch.rand((200,10000))
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()

print(toc-tic)
print(a.device)
print(b.device)

# 使用gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
a = torch.rand((10000,200),device = device) #可以指定在GPU上创建张量
b = torch.rand((200,10000)) #也可以在CPU上创建张量后移动到GPU上
b = b.to(device) #或者 b = b.cuda() if torch.cuda.is_available() else b 
tic = time.time()
c = torch.matmul(a,b)
toc = time.time()
print(toc-tic)
print(a.device)
print(b.device)


# 二，线性回归范例


# 准备数据
n = 1000000 #样本数量

X = 10*torch.rand([n,2])-5.0  #torch.rand是均匀分布 
w0 = torch.tensor([[2.0,-3.0]])
b0 = torch.tensor([[10.0]])
Y = X@w0.t() + b0 + torch.normal( 0.0,2.0,size = [n,1])  # @表示矩阵乘法,增加正态扰动

# 定义模型
class LinearRegression(nn.Module): 
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn_like(w0))
        self.b = nn.Parameter(torch.zeros_like(b0))
    #正向传播
    def forward(self,x): 
        return x@self.w.t() + self.b

linear = LinearRegression() 

# 训练模型
optimizer = torch.optim.Adam(linear.parameters(),lr = 0.1)
loss_func = nn.MSELoss()



def train(epoches):
    tic = time.time()
    for epoch in range(epoches):
        optimizer.zero_grad()
        Y_pred = linear(X) 
        loss = loss_func(Y_pred,Y)
        loss.backward() 
        optimizer.step()
        if epoch%50==0:
            print({"epoch":epoch,"loss":loss.item()})
    toc = time.time()
    print("time used:",toc-tic)


# CPU
train(500)

# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
linear.to(device)
X = X.to(device)
Y = Y.to(device)
train(500)