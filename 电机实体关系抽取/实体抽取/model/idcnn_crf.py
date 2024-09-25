# -*- coding:utf-8 -*-
import torch.nn as nn
from model import CRF
from torch.autograd import Variable
from model.cnn import IDCNN 
import torch


# 构建IDCNN_CRF模型核心类, 内部把IDCNN和CRF两个基础类进行了融合
class IDCNN_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, num_filters=64, dropout=0.4, use_cuda=True):
        super(IDCNN_CRF, self).__init__()
        # 重要参数初始化
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.use_cuda = use_cuda

        # 1: 设置词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2: 实例化IDCNN类, 输入维度是嵌入层的输出维度
        self.idcnn = IDCNN(input_size=embedding_dim, filters=num_filters)

        # 3: 对于CNN为基础的网络, dropout是标准配置
        self.dropout = nn.Dropout(p=dropout)
        
        # 4: 实例化CRF层, 本质上是一个参数方阵, 对应转移矩阵
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        
        # 5: 最后定义全连接映射层, 将卷积的输出维度映射到CRF层的输入维度上
        self.linear = nn.Linear(num_filters, tagset_size + 2)
    
    # 获取网络的输出特征张量
    def get_output_score(self, sentence, attention_mask=None):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)

        # 1: 执行初始化函数的第1层
        embeds = self.embedding(sentence)

        # 2: 执行初始化函数的第2层
        idcnn_out = self.idcnn(embeds, seq_length)

        # 3: 执行初始化函数的第3层
        idcnn_out = self.dropout(idcnn_out)
        
        # 4: 执行初始化函数的第5层
        out = self.linear(idcnn_out)
        
        # 5: 进行张量shape的转变, 保持经典的(batch_size, seq_length, X)模式
        feats = out.contiguous().view(batch_size, seq_length, -1)
        
        return feats

    # 当模型处于inference阶段时, 默认执行的前向计算函数
    def forward(self, sentence, masks):
        # 首先获取发射矩阵的输出张量
        feats = self.get_output_score(sentence)
        # 利用发射矩阵的输出张量, 直接进行维特比解码, 得到预测序列
        scores, tag_seq = self.crf._viterbi_decode(feats, masks.bool())
        return tag_seq

    # 当模型处于训练阶段时, 真正执行的前向计算函数, 为了得到最大似然损失值
    def neg_log_likelihood_loss(self, sentence, mask, tags):
        # 首先获取发射矩阵的输出张量
        feats = self.get_output_score(sentence)
        # 利用CRF层的最大似然损失函数计算当前损失值
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        # 计算当前batch的平均损失值
        batch_size = feats.size(0)
        loss_value /= float(batch_size)

        return loss_value

