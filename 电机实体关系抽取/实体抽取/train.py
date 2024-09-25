# -*- coding:utf-8 -*-

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils import load_vocab, load_data, recover_label, get_ner_fmeasure
from config import *
from model.idcnn_crf import IDCNN_CRF
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_cuda = True if torch.cuda.is_available() else False

vocab = load_vocab(vocab_file)
vocab_size = len(vocab)

# 读取训练集
print('max_length', max_length)
train_data = load_data(train_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
# print('train_data--->',train_data)
# 提取训练数据中的3个重要字段, 并封装成LongTensor类型
# 创建一个包含训练数据中所有输入id的LongTensor张量
train_ids = torch.LongTensor([temp.input_id for temp in train_data])
# 创建一个包含训练数据中所有输入mask的LongTensor张量
train_masks = torch.LongTensor([temp.input_mask for temp in train_data])
# 创建一个包含训练数据中所有标签id的LongTensor张量
train_tags = torch.LongTensor([temp.label_id for temp in train_data])
# 封装数据集 + 封装迭代器
train_dataset = TensorDataset(train_ids, train_masks, train_tags)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
# print('train_loader--->',train_loader)


# 读取测试集
dev_data = load_data(dev_file, max_length=max_length, label_dic=l2i_dic, vocab=vocab)
# print('dev_data--->',dev_data)
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)


# 测试函数
def evaluate(model, dev_loader):
    # 将模型设置为评估模式
    model.eval()

    # 初始化预测列表, 标签列表
    pred = []
    gold = []
    print('evaluate')
    # 循环遍历测试集, 并评估关键指标
    for i, dev_batch in enumerate(dev_loader):
        with torch.no_grad():
            sentence, masks, tags = dev_batch
            # 对数据进行Variable封装
            sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
            # 是否使用GPU进行加速推理
            if use_cuda:
                sentence = sentence.cuda()
                masks = masks.cuda()
                tags = tags.cuda()

            # 利用模型进行推理
            predict_tags = model(sentence, masks)
            # 将预测值和真实标签添加进结果列表中
            pred.extend([t for t in predict_tags.tolist()])
            gold.extend([t for t in tags.tolist()])

    # 将数字化标签映射回真实标签
    pred_label, gold_label = recover_label(pred, gold, l2i_dic, i2l_dic)
    # print('pred_label--->',pred_label)
    # print('gold_label--->',gold_label)
    # 计算关键指标
    acc, p, r, f = get_ner_fmeasure(gold_label, pred_label)
    print('acc:{}，p: {}，r: {}, f: {}'.format(acc, p, r, f))
    # 评估结束后, 将模型设置为训练模式
    model.train()

    
    return acc, p, r, f

# 实例化模型对象model
model = IDCNN_CRF(vocab_size, tagset_size, 300, 64, dropout=0.4, use_cuda=use_cuda)

if use_cuda:
    model = model.cuda()

model.train()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0005)
train_loss = []
val_f = []
val_acc = []
val_p = []
val_r = []
best_f = -100
# 双重for循环训练模型
for epoch in range(epochs):
    print('epoch: {}，train'.format(epoch))
    for i, train_batch in enumerate(tqdm(train_loader)):
        sentence, masks, tags = train_batch
        sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)

        if use_cuda:
            sentence = sentence.cuda()
            masks = masks.cuda()
            tags = tags.cuda()
        
        optimizer.zero_grad()
        # 训练时的损失值, 需要通过调用最大似然损失计算的函数, 而不是默认的forward函数
        loss = model.neg_log_likelihood_loss(sentence, masks, tags)
        # "老三样"
        loss.backward()
        optimizer.step()

    print('epoch: {}，loss: {}'.format(epoch, loss.item()))
    train_loss.append(loss.item())



    
    # 每训练完一个epoch, 对测试集进行一次评估
    acc, p, r, f = evaluate(model, dev_loader)

    val_acc.append(acc)
    val_f.append(f)
    val_p.append(p)
    val_r.append(r)
    # print('epoch: 每训练完一个{}epoch，对测试集进行一次评估，'.format(epoch))
    # print('acc:{}，p: {}，r: {}, f: {}'.format(acc, p, r, f))
    # 每当有更优的F1值时, 更新最优F1, 并保存模型状态字典
    if f > best_f:
        torch.save(model.state_dict(), save_model_path)
        best_f = f
# 保存训练好的模型到saved_model文件
# torch.save(model.state_dict(), '../idcnn/saved_model/model.pt')

plt.figure(figsize=(12, 7))
plt.subplot(231)
plt.plot(train_loss, label='train loss')
plt.legend()

plt.subplot(232)
plt.plot(val_acc, label='val acc')
plt.legend()

plt.subplot(233)
plt.plot(val_p, label='val p')
plt.legend()

plt.subplot(234)
plt.plot(val_r, label='val r')
plt.legend()

plt.subplot(235)
plt.plot(val_f, label='val f')
plt.legend()

plt.show()




