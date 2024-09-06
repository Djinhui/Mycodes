# -*- coding: utf-8 -*-
# @Time : 2022/3/9 17:11
# @Author : TuDaCheng
# @File : TextCNN.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Config(object):
    """模型配置参数"""
    def __init__(self, embedding):
        self.model_name = "TextCNN"
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
        self.filter_size = (2, 3, 4)
        self.num_filters = 256


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:  # 加载初始化好的预训练词/字嵌入矩阵  微调funetuning
            print(config.embedding_pretrained.size())
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            print(self.embedding)
        else:  # 否则随机初始化词/字嵌入矩阵 指定填充对应的索引
            self.embedding = nn.Embedding(config.num_vocab, config.embed, padding_idx=config.num_vocab - 1)

        # 不同大小卷积核对应的卷积操作
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_size]
        )

        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_size), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = x.squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        content = x[0]
        out = self.embedding(content)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out