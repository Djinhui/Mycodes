#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:TuDaCheng
# @file:LSTM_model.py
# @time:2023/03/22


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Config(object):
    """TextRCNN模型配置参数"""
    def __init__(self, embedding):
        self.model_name = "LSTM_model"
        self.train_path = "./datas/data/train.txt"
        self.dev_path = "./datas/data/test.txt"
        self.test_path = "./datas/data/test.txt"
        self.data_path = "./datas/data/all_data.txt"
        self.vocab_path = "./datas/data/vocab_dict.pkl"
        self.dataset_pkl = "./datas/data/dataset_pkl"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.class_list = [x.strip() for x in open("datas/data/class.txt", encoding="utf-8").readlines()]
        self.save_path = "./datas/save_model/" + self.model_name + ".ckpt"
        self.log_path = "./datas/log"
        self.padding_size = 100
        self.num_vocab = 0
        self.dropout = 0.5
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_vocab = 0  # 词表大小 在运行时赋值
        self.num_epochs = 300
        self.learning_rate = 1e-3
        self.batch_size = 128
        self.embedding_pretrained = torch.tensor(
            np.load(embedding)["embeddings"].astype('float32')) \
            if embedding != 'random' else None
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300  # 字向量维度
        self.hidden_size = 256  # lstm隐藏单元数
        self.num_layers = 2  # lstm层数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            print(config.embedding_pretrained.size())
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            print(self.embedding)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)

        # 单层双向lstm batch_size为第一维度
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers, bidirectional=True,
                            batch_first=True, dropout=config.dropout)

        self.maxpool = nn.MaxPool1d(config.padding_size)  # 沿着长度方向做全局最大池化

        # 输出层
        self.fc = nn.Linear(config.hidden_size*2 + config.embed, config.num_classes)

    def forward(self, x):
        content = x[0]  # [batch,seq_len]
        content = torch.clamp(input=content, min=0, max=2362)
        embed = self.embedding(content)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(embed)  # [batch_size,seq_len,hidden_size*2]
        out = torch.cat((embed, out), 2)  # 把词嵌入和lstm输出进行拼接 (batch,seq_len.embed+hidden_size*2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)  # [batch,embed+hidden_size*2,seq_len]
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out