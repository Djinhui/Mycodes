# -*- coding:utf-8 -*-
import sys
import torch
from model.idcnn_crf import IDCNN_CRF
from config import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import load_vocab, load_data, recover_label, get_ner_BMES

model_file = './saved_model/idcnn_crf.pt'

# 获取输入
inputs = open('./test_data/新建文本文档.txt','r',encoding='utf-8').read()
inputs = inputs.replace('\n','')
print(inputs)
# 将输入写入文件，文件格式为 每个字符+空格+标注
f = open('./data/input.txt', 'w', encoding='utf-8')
for each_word in list(inputs):
    f.write(each_word + ' O\n')
f.write('\n')
f.close()

# 加载模型，并设置为评估模式
model = IDCNN_CRF(21128, tagset_size, 300, use_cuda=False)
model.load_state_dict(torch.load(model_file, map_location='cpu'))
model.eval()

# 加载词汇表
vocab = load_vocab(vocab_file)

# 准备数据load_data() 函数读取输入文件，将输入句子转换成数字 ID 的形式，并将标签转换成数字 ID 的形式，
# 以便训练模型。max_length 是最大输入长度，l2i_dic 表示标签与数字 ID 的对应关系。将处理好的数据存储在 dev_data 中。
dev_data = load_data('./data/input.txt', max_length=max_length, label_dic=l2i_dic, vocab=vocab)
print('dev_data--->',dev_data)
# 将输入ID转换张量形式
dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
# 将mask转换成张量形式
dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
# 标签转化成张量形式
dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
# 构造数据集
dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
# 构建数据批次，每次训练前打断数据顺序，增加训练效果
dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=batch_size)

# 预测
pred = []  # 存储预测标签
gold = []  # 存放真实标签
# 循环遍历dev_batch的每个batch，对实体进行实体预测
for i, dev_batch in enumerate(dev_loader):
    sentence, masks, tags = dev_batch
    # 将输入句子、输入句子的掩码以及对应的标签封装在Variable对象中
    sentence, masks, tags = Variable(sentence), Variable(masks), Variable(tags)
    # 将输入句子、输入句子的掩码作为参数传递给模型。返回一个预测结果的张量
    predict_tags = model(sentence, masks)
    break
# 打印预测结果
print('predict_tags--->', predict_tags)

# 从预测和标记中恢复实体标签，如果失败则打印错误信息并退出
try:
    pred.extend([t for t in predict_tags.tolist()])
    gold.extend([t for t in tags.tolist()])
    # 使用字典将数字标签转换成字符串标签
    pred_label, gold_label = recover_label(pred, gold, l2i_dic, i2l_dic)
    # 提取预测标签，并去掉符号
    pred_label = pred_label[0][1:]
#如出现异常，提示输入错误，并退出程序
except:
    print('Input Error.')
    sys.exit(-1)

# 读取输入文件，将预测的实体标签写入输出文件
in_file = open('./data/input.txt', 'r', encoding='utf-8')
out_file = open('./data/output.txt', 'w', encoding='utf-8')
# 读取输入文件中的所有行
lines = in_file.readlines()
# 存储预测标签后的新行
new_lines = []
# 循环遍历每一行，并将预测标签添加到该行的前两个字符
for index in range(len(lines)):
    if len(lines[index]) > 2:
        new_lines.append(lines[index][:2] + pred_label[index] + '\n')
# 将新行写入到输出文件中
out_file.writelines(new_lines)
# 关闭输入输出文件
in_file.close()
out_file.close()
# 打印预测结果
print('new_lines:', new_lines)

# 解析实体
all = []
index = 0
print('len newlines',len(new_lines))
# 在索引不等于new_lines的长度之前执行while循环
while index != len(new_lines):
    # 判断是否是命名实体识别开头
    print('index:', index)
    if new_lines[index][2] == 'B':
        # 从命名实体中提取标记
        tag = new_lines[index][4:7]
        print('tag--->',tag)
        # 循环遍历实体后的标记
        for j in range(index + 1, len(new_lines)):
            # 检查当前标记是否是命名实体的一部分
            if new_lines[j][2] == 'I':
                pass
            # 检查当前标记是否不是任何命名实体的一部分或表示新命名实体的开始
            elif new_lines[j][2] == 'O' or new_lines[j][2] == 'B' or new_lines[j][2] == 'S':
                index = j + 1
                break
            # 如果当前标记是命名实体的结尾，则提取并储存该实体
            else:
                print(str([index, j]))
                print(inputs[index: j + 1])
                all.append((tag, inputs[index: j + 1]))
                index = j + 1
                break
    else:
        index = index + 1

# 打印解析结果
print('all--->',all)
print('done')

with open(file='./data/motor_entity.csv',mode='w',encoding='utf-8') as f:
    f.write(''.join(map(str,all)))


