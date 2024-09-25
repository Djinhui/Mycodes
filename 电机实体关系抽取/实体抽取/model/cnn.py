# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from config import *


# 构建IDCNN核心类的代码
class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        # 采用原始论文中经典的4个Block, 每个Block中3层卷积的结构
        self.layers = [{'dilation': 1}, {'dilation': 1}, {'dilation': 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(256) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(256) for _ in range(num_block)])
        
        # 依次构建每一层网络
        for i in range(len(self.layers)):
            dilation = self.layers[i]['dilation']
            # 网络中的第一层结构是nn.Conv1d, 其中采用空洞卷积的方法
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            
            # 每一层网络都包含卷积层, 激活层, 正则化层, 依次添加进net中
            net.add_module('layer%d'%i, single_block)
            net.add_module('relu', nn.ReLU())
            net.add_module('layernorm', norms_1[i])

        # 最后定义一个全连接层
        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()

        # 依次构建4个Block, 每一个Block的内部结构都相同(只有微小的参数差别dilation)
        for i in range(num_block):
            self.idcnn.add_module('block%i' % i, net)
            self.idcnn.add_module('relu', nn.ReLU())
            self.idcnn.add_module('layernorm', norms_2[i])

    # 前向传播函数
    def forward(self, embeddings, length):
        # 1: 首先对词嵌入张量进行全连接映射的转换
        embeddings = self.linear(embeddings)
        # 2: 调整第1, 2维度
        embeddings = embeddings.permute(0, 2, 1)
        # 3: 最后进行IDCNN的特征提取并将第2步的维度调整回原状
        output = self.idcnn(embeddings).permute(0, 2, 1)

        return output


# 采用经典transformer的LayerNorm实现策略
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # 初始化一个全1的张量, 初始化一个全0的张量
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 计算得到均值和方差, 注意是在当前样本的横向维度, 以区别于BatchNorm
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 按照公式计算, 并返回结果
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

