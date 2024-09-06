# -*- coding: utf-8 -*-
# @Time : 2022/11/16 11:05
# @Author : TuDaCheng
# @File : train.py
import os
import torch
import pickle as pkl
from libs.utils import Logger
from torch.nn import BCEWithLogitsLoss
from libs.metric import Metric
from transformers import AdamW, get_linear_schedule_with_warmup
from libs.loss import MultiLabelBalancedLoss

logger = Logger(os.path.join("./datas/log", "Bert_log.txt"))

metric = Metric()
# criterion = BCEWithLogitsLoss()  # 自定义多标签损失函数 （多标签平衡损失）
criterion = MultiLabelBalancedLoss()  # 自定义多标签损失函数 （多标签平衡损失）


def train(config, model, train_iter, dev_iter, test_iter):
    logger.log()
    logger.log("model_name:", config.model_name)
    logger.log("Device:", config.device)
    logger.log("Epochs:", config.num_epochs)
    logger.log("Batch Size:", config.batch_size)
    logger.log("Learning Rate:", config.learning_rate)
    logger.log("Max Sequence Length:", config.pad_size)
    logger.log()

    model.train()

    # 不需要衰减的参数
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

    total_step = len(train_iter) * config.num_epochs
    num_warmup_steps = round(total_step * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_step)
    model.to(config.device)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0  # 记录进行到多少batch
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        step = 0
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            step += 1

            outputs = model(trains)
            model.zero_grad()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率衰减

            if total_batch % 20 == 0:
                s = "Train Epoch: {:d} Step: {:d} Loss: {:.4f}".format(epoch, step, loss.item())
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

    with open("datas/data/label.pkl", "rb") as f:
        need_labels = pkl.load(f)
    # print each class
    for label, precision, recall, f1 in zip(need_labels, precisions, recalls, f1s):
        s = "Precision: {:.4f} Recall: {:.4f} F1: {:.4f} Class: {:s}".format(precision, recall, f1, label)
        logger.log(s)

    logger.log()
    s = "Micro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(micro_precision, micro_recall, micro_f1)
    logger.log(s)

    s = "Macro Average - Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(macro_precision, macro_recall, macro_f1)
    logger.log(s)

