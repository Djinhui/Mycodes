# -*- coding: utf-8 -*-
# @Time : 2023/2/6 9:57
# @Author : TuDaCheng
# @File : Bert.py
import json
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Config(object):
    """
    配置参数类
    """
    def __init__(self, dataset):

        self.model_name = "Bert"
        self.train_path = dataset + '/train.json'  # 训练集
        self.dev_path = dataset + '/dev.json'  # 验证集
        self.test_path = dataset + '/test.json'  # 测试集
        self.label_path = dataset + '/label.pkl'  # 标签集

        # 构建数据集pkl文件
        self.datasetpkl = dataset + "/datasetBert.pkl"
        # 类别
        self.class_list = [x.strip() for x in open(dataset + "/class.txt", "r", encoding="utf-8").readlines()]
        # 模型训练结果
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 若超过1000个batch效果还没有提升， 提前终止训练
        self.require_improvement = 1000

        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 50
        # batch_size
        self.batch_size = 32
        # 每句话处理长度（短填，长切）
        self.pad_size = 300
        # 学习率
        self.learning_rate = 5e-5
        # bert预训练模型位置
        self.bert_path = "./bert-base-chinese"
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐藏层数
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True  # 微调时一定要用True  否则效果会很差
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [input_ids, seq_length, mask]
        token_ids = x[0]  # 对应输入的句子 shape [batch_size, seq_length]---->[128, 32]
        masks = x[2]  # 对padding部分进行mask shpe [128, 32]
        segments_id = x[3]
        # _, pooled_output = self.bert(token_ids, attention_mask=masks, output_all_encoded_layers=False)  # shape [batch_size, hidden_size]--->[128,768]
        _, pooled_output, = self.bert(token_ids, attention_mask=masks, token_type_ids=segments_id, return_dict=False)  # shape [batch_size, hidden_size]--->[128,768]
        y = self.dropout(pooled_output)
        out = self.fc(pooled_output)  # shape [128, 10]
        # out = torch.sigmoid(self.fc(pooled_output))
        return out