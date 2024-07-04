import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from sklearn import datasets
from torchkeras import Model, summary  # 仿keras的torch

'''
Dataset定义了数据集的内容,它相当于一个类似列表的数据结构,具有确定的长度,能够用索引获取数据集中的元素
DataLoader定义了按batch加载数据集的方法,它是一个实现了__iter__方法的可迭代对象,每次迭代输出一个batch的数据
用户只需实现Dataset的__len__方法和__getitem__方法,就可以轻松构建自己的数据集,并用默认数据管道进行加载

DataLoader(Dataset())

DataLoader能够控制batch的大小,batch中元素的采样方法,
以及将batch结果整理成模型所需输入形式的方法,并且能够使用多进程读取数据
'''

# 一、 使用Dataset创建数据集
'''
使用 torch.utils.data.TensorDataset 根据Tensor创建数据集(numpy的array,Pandas的DataFrame需要先转换成Tensor)。
使用 torchvision.datasets.ImageFolder 根据图片目录创建图片数据集。
继承 torch.utils.data.Dataset 创建自定义数据集
通过torch.utils.data.random_split 将一个数据集分割成多份，常用于分割训练集，验证集和测试集
调用Dataset的加法运算符(+)将多个数据集合并成一个数据集
'''
# 1. 根据Tensor创建数据集
iris = datasets.load_iris()
ds_iris = TensorDataset(torch.tensor(iris.data), torch.tensor(iris.target))

n_train = int(len(ds_iris) * 0.8)
n_valid = len(ds_iris) - n_train
ds_train, ds_valid = random_split(ds_iris,[n_train, n_valid])

# 演示加法运算符（`+`）的合并作用
ds_data = ds_train + ds_valid
print('len(ds_train) = ',len(ds_train))
print('len(ds_valid) = ',len(ds_valid))
print('len(ds_train+ds_valid) = ',len(ds_data))
print(type(ds_data))

# 使用DataLoader加载数据集
dl_train, dl_valid = DataLoader(ds_train, batch_size=8), DataLoader(ds_valid, batch_size=8)
for features,labels in dl_train:
    print(features,labels)

# 2 根据图片目录创建图片数据集
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 
from PIL import Image

img = Image.open('../data/cat.jpeg')
# 随机数值翻转
transforms.RandomVerticalFlip()(img)
#随机旋转
transforms.RandomRotation(45)(img)
# 图片增强
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(), #随机水平翻转
    transforms.RandomVerticalFlip(), #随机垂直翻转
    transforms.RandomRotation(45),  #随机在45度角度内旋转
    transforms.ToTensor() #转换成张量
])

transform_valid = transforms.Compose([
    transforms.ToTensor()
  ]
)

# 根据图片目录创建数据集
ds_train = datasets.ImageFolder("../data/cifar2/train/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())
ds_valid = datasets.ImageFolder("../data/cifar2/test/",
            transform = transform_train,target_transform= lambda t:torch.tensor([t]).float())

print(ds_train.class_to_idx)

# 使用DataLoader加载数据集
dl_train = DataLoader(ds_train,batch_size = 50,shuffle = True,num_workers=3)
dl_valid = DataLoader(ds_valid,batch_size = 50,shuffle = True,num_workers=3)

for features,labels in dl_train:
    print(features.shape)
    print(labels.shape)
    break

# 3，创建自定义数据集
# 通过继承Dataset类创建imdb文本分类任务的自定义数据集
import numpy as np 
import pandas as pd 
from collections import OrderedDict
import re,string
import os

MAX_WORDS = 10000 # 仅考虑最高频的10000词
MAX_LEN = 200 # 每个样本保留200个词的长度
BATCH_SIZE = 20

train_data_path = 'data/imdb/train.tsv'
test_data_path = 'data/imdb/test.tsv'
train_token_path = 'data/imdb/train_token.tsv'
test_token_path =  'data/imdb/test_token.tsv'
train_samples_path = 'data/imdb/train_samples/'
test_samples_path =  'data/imdb/test_samples/'

# 构建词典
word_count_dict = {}
# 清洗文本
def clean_text(text):
    lowercase = text.lower().replace('\n', ' ')
    stripped_html = re.sub('<br />', ' ', lowercase)
    cleaned_punctuation = re.sub('[%s]' %re.escape(string.punctuation), '', stripped_html)
    return cleaned_punctuation

with open(train_data_path, 'r', encoding='utf-8') as f:
    for line in f:
        label, text = line.split('\t')
        cleaned_text = clean_text(text)
        for word in clean_text.split(' '):
            word_count_dict[word] = word_count_dict.get(word, 0) + 1

df_word_dict = pd.DataFrame(pd.Series(word_count_dict, name='count'))
df_word_dict = df_word_dict.sort_values(by='count', ascending=False)

 # 编号0和1分别留给未知词<unkown>和填充<padding>
df_word_dict = df_word_dict[0:MAX_WORDS-2]
df_word_dict['word_id'] = range(2, MAX_WORDS)
word_id_dict = df_word_dict['word_id'].to_dict()
# df_word_dict.head()
# 转换token
# 填充文本
def pad(data_list, pad_length):
    padded_list = data_list.copy()
    if len(data_list) > pad_length:
        padded_list = data_list[-pad_length:]
    if len(data_list) < pad_length:
        padded_list = [1] * (pad_length - len(data_list)) + data_list
    return padded_list

def text_to_token(text_file, token_file):
    with open(text_file, 'r', encoding='utf-8') as fin, open(token_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            label, text = line.split('\t')
            cleaned_text = clean_text(text)
            word_token_list = [word_id_dict.get(word, 0) for word in cleaned_text.split(" ")]
            pad_list = pad(word_token_list, MAX_LEN)
            out_line = label + '\t' + ' '.join([str(x) for x in pad_list])
            fout.write(out_line+'\n')

text_to_token(train_data_path,train_token_path)
text_to_token(test_data_path,test_token_path)

if not os.path.exists(train_samples_path):
    os.mkdir(train_samples_path)

if not os.path.exists(test_samples_path):
    os.mkdir(test_samples_path)

def split_samples(token_path, samples_dir):
    with open(token_path, 'r', encoding='utf-8') as fin:
        i = 0
        for line in fin:
            with open(samples_dir + '%d.txt'%i, 'w', encoding='utf-8') as fout:
                fout.write(line)
            i += 1

split_samples(train_token_path,train_samples_path)
split_samples(test_token_path,test_samples_path)
print(os.listdir(train_samples_path)[0:100])

class ImdbDataset(Dataset):
    def __int__(self, samples_dir):
        self.samples_dir = samples_dir
        self.samples_paths = os.listdir(samples_dir)

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self, index):
        path = self.samples_dir + self.samples_paths[index]
        with open(path, 'r', encoding='utf-8') as f:
            line = f.readline()
            label, tokens = line.split('t')
            label = torch.tensor([float(label)], dtype=torch.float)
            feature = torch.tensor([int(x) for x in tokens.split(' ')], dtype=torch.long)
            return (feature, label)

ds_train = ImdbDataset(train_samples_path)
ds_test = ImdbDataset(test_samples_path)

print(len(ds_train))
print(len(ds_test))

# 加载
dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dl_test = DataLoader(ds_test,batch_size = BATCH_SIZE,num_workers=4)

for features,labels in dl_train:
    print(features)
    print(labels)
    break

'''
tensor([[   1,    1,    1,  ...,   29,    8,    8],
        [  13,   11,  247,  ...,    0,    0,    8],
        [8587,  555,   12,  ...,    3,    0,    8],
        ...,
        [   1,    1,    1,  ...,    2,    0,    8],
        [ 618,   62,   25,  ...,   20,  204,    8],
        [   1,    1,    1,  ...,   71,   85,    8]])
tensor([[1.],
        [0.],
        [0.],
        ...,
        [1.],
        [0.],
        [1.]])
'''

from torchkeras import Model,summary

class Net(Model):

    def __init__(self) -> None:
        super(Net, self).__init__()
        #设置padding_idx参数后将在训练过程中将填充的token始终赋值为0向量
        self.embedding = nn.Embedding(num_embeddings = MAX_WORDS,embedding_dim = 3,padding_idx = 1)
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1",nn.Conv1d(in_channels = 3,out_channels = 16,kernel_size = 5))
        self.conv.add_module("pool_1",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_1",nn.ReLU())
        self.conv.add_module("conv_2",nn.Conv1d(in_channels = 16,out_channels = 128,kernel_size = 2))
        self.conv.add_module("pool_2",nn.MaxPool1d(kernel_size = 2))
        self.conv.add_module("relu_2",nn.ReLU())

        self.dense = nn.Sequential()
        self.dense.add_module("flatten",nn.Flatten())
        self.dense.add_module("linear",nn.Linear(6144,1))
        self.dense.add_module("sigmoid",nn.Sigmoid())

    def forward(self,x):
        x = self.embedding(x).transpose(1,2)
        x = self.conv(x)
        y = self.dense(x)
        return y

model = Net()
print(model)

model.summary(input_shape = (200,),input_dtype = torch.LongTensor)

def accuracy(y_pred, y_true):
    y_pred = torch.where(y_pred > 0.5, torch.ones_like(y_pred, dtype=torch.float32),\
        torch.zeros_like(y_pred, dtype=torch.float32))
    acc = torch.mean(1 - torch.abs(y_true - y_pred))
    return acc

model.compile(loss_func=nn.BCELoss(), optimizer=torch.optim.Adagrad(model.parameters(), lr=.2),metrics_dict={"accuracy":accuracy})
# 训练模型
dfhistory = model.fit(10,dl_train,dl_val=dl_test,log_step_freq= 200)


# 二、使用DataLoader加载数据集
# DataLoader能够控制batch的大小，batch中元素的采样方法，
# 以及将batch结果整理成模型所需输入形式的方法，并且能够使用多进程读取数据
# 一般情况下，我们仅配置 dataset, batch_size, shuffle, num_workers, drop_last这五个参数，其他参数使用默认值即可
'''DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    timeout=0,
    worker_init_fn=None,
    multiprocessing_context=None,
)'''

#构建输入数据管道
ds = TensorDataset(torch.arange(1,50))
dl = DataLoader(ds,
                batch_size = 10,
                shuffle= True,
                num_workers=2,
                drop_last = True)
#迭代数据
for batch, in dl:
    print(batch)