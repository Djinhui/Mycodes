# coding:utf-8

import torch
from model import *
from data_loader import *
import pandas as pd
from tqdm import tqdm
from utils import *
import json
from config import *
import matplotlib.pyplot as plt

def model2train(model,train_iter,dev_iter,optimizer,conf):
    epochs = conf.epochs
    best_triple_f1 = 0
    for epoch in range(epochs):
        train_epoch(model,train_iter,dev_iter,optimizer,best_triple_f1,epoch)

    torch.save(model.state_dict(), '../save_model/last_model.pth')

sub_precision = []
sub_recall = []
sub_f1 = []
triple_precision = []
triple_recall = []
triple_f1 = []
train_loss = []

def train_epoch(model,train_iter,dev_iter,optimizer,best_triple_f1,epoch):
    for step,(inputs,labels) in enumerate(tqdm(train_iter)):
        print('step--->',step)
        model.train()
        logist = model(**inputs)
        loss = model.compute_loss(**logist,**labels)
        print('得出的损失是',loss)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            torch.save(model.state_dict(),'../save_model/epoch_%s_model_%s.pth' % (epoch, step))
            results = dev(model,dev_iter)
            print(results[-1])
            if results[-2] > best_triple_f1:
                best_triple_f1 = results[-2]
                torch.save(model.state_dict(),'../save_model/best_new_f1.pth')
                print('epoch:{},'
                      'step:{},'
                      'sub_precision:{:.4f}, '
                      'sub_recall:{:.4f}, '
                      'sub_f1:{:.4f}, '
                      'triple_precision:{:.4f}, '
                      'triple_recall:{:.4f}, '
                      'triple_f1:{:.4f},'
                      'train loss:{:.4f}'.format(epoch,
                                                 step,
                                                 results[0],
                                                 results[1],
                                                 results[2],
                                                 results[3],
                                                 results[4],
                                                 results[5],
                                                 loss.item()))
                sub_precision.append(results[0])
                sub_recall.append(results[1])
                sub_f1.append(results[2])
                triple_precision.append(results[3])
                triple_recall.append(results[4])
                triple_f1.append(results[5])
                train_loss.append(loss.item())

    return best_triple_f1

def plot_figure():
    plt.figure(figsize=(12, 7))
    plt.subplot(241)
    plt.plot(sub_precision, label='sub_precision')
    plt.legend()

    plt.subplot(242)
    plt.plot(sub_recall, label='sub_recall')
    plt.legend()

    plt.subplot(243)
    plt.plot(sub_f1, label='sub_f1')
    plt.legend()

    plt.subplot(244)
    plt.plot(triple_precision, label='triple_precision')
    plt.legend()

    plt.subplot(245)
    plt.plot(triple_recall, label='triple_recall')
    plt.legend()

    plt.subplot(246)
    plt.plot(triple_f1, label='triple_f1')
    plt.legend()

    plt.subplot(247)
    plt.plot(train_loss, label='train_loss')
    plt.legend()

    plt.savefig('my_plot.png')
    plt.show()

# 验证模型效果
def dev(model,dev_iter):
    # 将模型设置为评估模式
    model.eval()
    df = pd.DataFrame(columns=['TP','PRED','REAL','p','r','f1'],index=['sub','triple'])
    df.fillna(0, inplace=True)
    for inputs,labels in tqdm(dev_iter):
        # 通过模型得到预测结果
        logist = model(**inputs)
        # 将预测结果转换成0或1的形式
        pred_sub_heads = convert_score_to_zero_one(logist['pred_sub_heads'])
        pred_sub_tails = convert_score_to_zero_one(logist['pred_sub_tails'])
        sub_heads = convert_score_to_zero_one(labels['sub_heads'])
        sub_tails = convert_score_to_zero_one(labels['sub_tails'])
        # 获取每一批次数据的大小
        batch_size = inputs['input_ids'].shape[0]
        obj_heads = convert_score_to_zero_one(labels['obj_heads'])
        obj_tails = convert_score_to_zero_one(labels['obj_tails'])
        pred_obj_heads = convert_score_to_zero_one(logist['pred_obj_heads'])
        pred_obj_tails = convert_score_to_zero_one(logist['pred_obj_tails'])

        for batch_index in range(batch_size):
            # 提取预测和真实主体
            pred_subs = extract_sub(pred_sub_heads[batch_index].squeeze(),pred_sub_tails[batch_index].squeeze())
            true_subs = extract_sub(sub_heads[batch_index].squeeze(),sub_tails[batch_index].squeeze())
            # 提取预测和真实三元组
            pred_ojbs = extract_obj_and_rel(pred_obj_heads[batch_index],pred_obj_tails[batch_index])
            true_objs = extract_obj_and_rel(obj_heads[batch_index],obj_tails[batch_index])
            # 更新数据框，记录预测和真实主体的数量
            df['PRED']['sub'] += len(pred_subs)
            df['REAL']['sub'] += len(true_subs)
            # 计算主体提取任务中的TP数量
            for true_sub in true_subs:
                if true_sub in pred_subs:
                    df['TP']['sub'] += 1
            # 更新数据框，记录预测和真实三元组的数量
            df['PRED']['triple'] += len(pred_ojbs)
            df['REAL']['triple'] += len(true_objs)
            # 计算三元组中的TP数量
            for true_obj in true_objs:
                if true_obj in pred_ojbs:
                    df['TP']['triple'] += 1

    # 计算主体提取任务的精度、召回率和F1得分
    df.loc['sub','p'] = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    df.loc['sub','r'] = df['TP']['sub'] / (df['REAL']['sub'] + 1e-9)
    df.loc['sub','f1'] = 2 * df['p']['sub'] * df['r']['sub'] / (df['p']['sub'] + df['r']['sub'] + 1e-9)
    sub_precision = df['TP']['sub'] / (df['PRED']['sub'] + 1e-9)
    sub_recall = df['TP']['sub'] /(df['REAL']['sub'] + 1e-9)
    sub_f1 = 2 * sub_precision * sub_recall / (sub_precision + sub_recall + 1e-9)

    # 计算三元组的提取任务的精度、召回率、F1
    df.loc['triple','p'] = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    df.loc['triple','r'] = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    df.loc['triple','f1'] = 2 * df['p']['triple'] * df['r']['triple'] / (
        df['p']['triple'] + df['r']['triple'] + 1e-9
    )

    triple_precision = df['TP']['triple'] / (df['PRED']['triple'] + 1e-9)
    triple_recall = df['TP']['triple'] / (df['REAL']['triple'] + 1e-9)
    triple_f1 = 2 * triple_precision * triple_recall / (
        triple_precision + triple_recall + 1e-9
    )




    # 返回主体提取任务和三元组提取任务的精度、召回率和F1得分，以及更新后的数据框
    return sub_precision,sub_recall,sub_f1,triple_precision,triple_recall,triple_f1,df



if __name__ == '__main__':
    conf = Config()
    model,optimizer,sheduler,device = load_model(conf)
    train_iter, dev_iter = get_loader()
    model2train(model,train_iter,dev_iter,optimizer,conf)
    plot_figure()
