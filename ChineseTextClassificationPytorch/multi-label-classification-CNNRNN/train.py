# -*- coding: utf-8 -*-
# @Time : 2022/11/16 11:05
# @Author : TuDaCheng
# @File : train.py
import os
import torch
import pickle as pkl
import torch.nn as nn
from libs.utils import Logger
from torch.nn import BCEWithLogitsLoss
from libs.metric import Metric
from libs.loss import MultiLabelBalancedLoss

logger = Logger(os.path.join("./datas/log", "log.txt"))

metric = Metric()
criterion = BCEWithLogitsLoss()  # 多标签的BCEWithLogitsLoss()
# criterion = MultiLabelBalancedLoss()  # 自定义多标签损失函数 （多标签平衡损失）


# 基于方差缩放的参数初始化
# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                elif method == "orthogonal":  ## 正交初始化
                    nn.init.orthogonal(w, gain=1)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("dropout", config.dropout)
    logger.log("Max Sequence Length:", config.pad_size)
    logger.log()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        step = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            step += 1
            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            if total_batch % 20 == 0:
                s = "Train Epoch: {:d} Step: {:d} Loss: {:.6f}".format(epoch, step, loss.item())
                logger.log(s)

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                precision, recall, f1, improved, stop = evaluate(model, dev_iter)

                s = "Eval Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(precision, recall, f1)
                logger.log(s)

                if improved:
                    torch.save(model.state_dict(), config.save_path)
                    logger.log("Improved! Best F1: {:.4f}".format(f1))
                    logger.log()

                if stop:
                    logger.log("STOP! NO PATIENT!")
                    flag = True
                    break

                model.train()
            total_batch += 1
            # if total_batch - last_improve > config.require_improvement:
            #     # 验证集loss超过1000batch没下降，结束训练
            #     print("No optimization for a long time, auto-stopping...")
            #     flag = True
            #     break
        if flag:
            break
    test(config, model, test_iter)


def evaluate(model, data_iter):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss_total += loss

            truth = labels.tolist()
            pred = outputs.sigmoid().cpu().tolist()

            metric.store(pred, truth)

    # 获得微平均指标
    _, _, (precision, recall, f1) = metric.precision_recall_f1()

    improved, stop = metric.is_improved(f1)

    return precision, recall, f1, improved, stop


def test(config, model, test_iter):
    # test
    logger_test = Logger(os.path.join(config.log_path, "log-test.txt"))

    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    loss_total = 0
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts)
            labels = labels.float()
            loss = criterion(outputs, labels)
            loss_total += loss

            truth = labels.tolist()
            pred = outputs.sigmoid().cpu().tolist()

            metric.store(pred, truth)

        (precisions, recalls, f1s), (macro_precision, macro_recall, macro_f1), (micro_precision, micro_recall, micro_f1) \
            = metric.precision_recall_f1(threshold=0.5)

    need_labels = config.class_list
    # print each class
    for label, precision, recall, f1 in zip(need_labels, precisions, recalls, f1s):
        s = "Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Class: {:s}".format(precision, recall, f1, label)
        logger_test.log(s)

    logger_test.log()
    s = "Micro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(micro_precision, micro_recall, micro_f1)
    logger_test.log(s)

    s = "Macro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(macro_precision, macro_recall, macro_f1)
    logger_test.log(s)

