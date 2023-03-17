import torch
import torch.nn as nn
import torchvision

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
        self.dense.add_module('fc', nn.Linear(6144, 1))
        self.dense.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y

net = Net()

print('======================================child==============================')
i = 0
for child in net.children():
    i += 1
    print(child, '\n')
print("child number",i)
print('===========================================================================')

print('==============================named_children==============================')
i = 0
for name,child in net.named_children():
    i+=1
    print(name,":",child,"\n")
print("child number",i) 
print('===========================================================================')

print('======================================modules==============================')
i = 0
for child in net.modules():
    i += 1
    print(child, '\n')
print("module number",i)
print('===========================================================================')


print('================================named_modules==============================')
i = 0
for name,child in net.named_modules():
    i+=1
    print(name,":",child,"\n")
print("module number",i) 
print('===========================================================================')


print('================================named_parameters==============================')
for k, v in net.named_parameters():
    print(k)


print('=============================')
print([p for p in net.dense.fc.parameters()])
# print(net.dense.fc.requires_grad) error
print(net.dense.fc.bias)
print(net.dense.fc.in_features)