#!/usr/bin/python
# -*- coding: UTF-8 -*-
# @author:TuDaCheng
# @file:run.py
# @time:2023/03/22


import torch
import time
import argparse
import numpy as np
from importlib import import_module
from utils.data_utils import build_dataset, build_iterator, get_time_dif
from train import train
# 参数配置
parser = argparse.ArgumentParser(description="Chinese Text Classification")  # 声明argparse对象 可附加说明

# 添加模型参数 模型是必须设置的参数(required=True) 类型是字符串
parser.add_argument("--model", type=str, default="TextCNN", help="choose a model: TextCNN, TextRNN")

# embedding随机初始化或使用预训练词或字向量 默认使用预训练
parser.add_argument("--embedding", default="pre_trained", type=str, help="random or pre_trained")
# 基于词还是基于字 默认基于字
parser.add_argument("--word", default=False, type=bool, help="True for word, False for char")

# 解析参数
args = parser.parse_args()

if __name__ == '__main__':

    print(torch.cuda.is_available())
    model_name = args.model
    x = import_module("models." + model_name)  # 根据所选模型名字在models包下 获取相应模块(.py)
    embedding = "datas/data/embedding.npz"
    if args.embedding == 'random':
        embedding = 'random'

    config = x.Config(embedding)
    # 设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    start_time = time.time()

    # 数据预处理
    vocab, train_data, dev_data, test_data = build_dataset(config)  # 构建词典、训练集、验证集、测试集
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("模型开始之前 准备时间：", time_dif)

    config.num_vocab = len(vocab)
    model = x.Model(config).to(config.device)

    train(config, model, train_iter, dev_iter, test_iter)
