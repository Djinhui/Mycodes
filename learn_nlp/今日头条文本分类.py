# -*- coding: utf-8 -*-
# from:https://github.com/frank330/Text_Classification CNN-LSTM  TextCNN
# 文件格式：
# 6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
# 每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词




# 1. ------------------------------------CNN-LSTM----------------------------------
import numpy as np
import pandas  as pd
from jieba import lcut
import matplotlib
import matplotlib.pyplot as plt
import PySimpleGUI as sg
from sklearn.model_selection import train_test_splitrt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv1D, MaxPooling1D, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping, TensorBoard

# 数据处理
# 判断一个unicode是否是汉字
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False
    
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str  

def getStopWords():
    file = open('./data/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words


def dataParse(text:str, stop_words):
    label_map = {'news_story': 0, 'news_culture': 1, 'news_entertainment': 2,
               'news_sports': 3, 'news_finance': 4, 'news_house': 5, 'news_car': 6,
               'news_edu': 7, 'news_tech': 8, 'news_military': 9, 'news_travel': 10,
               'news_world': 11, 'stock': 12, 'news_agriculture': 13, 'news_game': 14}
    _, _, label,content,_ = text.split('_!_')
    label = label_map[label]
    content = reserve_chinese(content)  
    words = lcut(content)
    words = [word for word in words if not word in stop_words]
    return words, int(label)

def getData(file='./data/toutiao_cat_data.txt'):
    file = open(file, 'r',encoding='utf8')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = [] # list of lists of strings
    al_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        al_labels.append(label)
    return all_words, al_labels

data, label = getData()

X_train, X_t, train_y, v_y = train_test_splitrt(data, label, test_size=0.2, random_state=42)
X_val, X_test, val_y, test_y = train_test_splitrt(X_t, v_y, test_size=0.5, random_state=42)

# 对标签数据进行one-hot编码
ohe = OneHotEncoder()
train_y = ohe.fit_transform(np.array(train_y).reshape(-1, 1)).toarray()
val_y = ohe.fit_transform(np.array(val_y).reshape(-1, 1)).toarray()
test_y = ohe.fit_transform(np.array(test_y).reshape(-1, 1)).toarray()

## 使用Tokenizer对词组进行编码
## 当我们创建了一个Tokenizer对象后，使用该对象的fit_on_texts()函数，以空格去识别每个词,
## 可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
max_words = 5000
max_len = 100
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(data) # data can be 1. a list of strings or  2. a list of list of string, here data is format type 2. 如下所示

'''
tokens_samples = ['爸爸 妈妈 爱 我', '爸爸 妈妈 爱 中国', '我 爱 中国']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens_samples)
word_index = tokenizer.word_index
print(word_index)  # {'爱': 1, '爸爸': 2, '妈妈': 3, '我': 4, '中国': 5}
print(len(word_index))  # 5

tokens_samples = [['爸爸', '妈妈', '爱', '我'], ['爸爸', '妈妈', '爱', '中国'], ['我','爱', '中国']]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens_samples)
print(word_index)   # {'爱': 1, '爸爸': 2, '妈妈': 3, '我': 4, '中国': 5}
print(len(word_index))   # 5

但是
tokens_samples = [['爸爸 妈妈 爱 我'], ['爸爸 妈妈 爱 中国'], ['我 爱 中国']]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens_samples)
print(word_index)   # {'爸爸 妈妈 爱 我': 1, '爸爸 妈妈 爱 中国': 2, '我 爱 中国': 3}
print(len(word_index))  # 3
'''

# texts_to_sequences 输出的是根据对应关系输出的向量序列 token-id，是不定长的，跟句子的长度有关系
train_seq = tok.texts_to_sequences(X_train)
val_seq = tok.texts_to_sequences(X_val)
test_seq = tok.texts_to_sequences(X_test)

# 使用pad_sequences函数将序列填充为相同的长度
train_seq = pad_sequences(train_seq, maxlen=max_len)
val_seq = pad_sequences(val_seq, maxlen=max_len)
test_seq = pad_sequences(test_seq, maxlen=max_len)

num_classes = 15

# 定义CNN-LSTM模型

inputs = Input(shape=[max_len], name='inputs')
## Embedding(词汇表大小,batch大小,每个新闻的词长)
layer = Embedding(max_words+1, 128, input_length=max_len)(inputs)
layer = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(layer) # (None, 100, 32)
layer = MaxPooling1D(pool_size=2)(layer) # (None, 50, 32)
layer = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(layer) # (None, 50, 32) 
layer = MaxPooling1D(pool_size=2)(layer) # (None, 25, 32) 
layer = LSTM(units=128, dropout=0.2, recurrent_dropout=0.2)(layer) # (None, 128)  
layer = Dense(units=num_classes, activation='softmax')(layer) # (None, 15)
model = Model(inputs=inputs, outputs=layer)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_seq, train_y, batch_size=128, epochs=10, validation_data=(val_seq, val_y),
          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001),TensorBoard(log_dir='./logs')])

# 保存模型
model.save('./model/CNN-LSTM.h5')
del model

# 加载模型
model = load_model('./model/CNN-LSTM.h5')

test_pre = model.predict(test_seq)
pred = np.argmax(test_pre, axis=1)
real = np.argmax(test_y, axis=1)

cv_conf = confusion_matrix(real, pred)
acc = accuracy_score(real, pred)
precision = precision_score(real, pred, average='macro')
recall = recall_score(real, pred, average='macro')
f1 = f1_score(real, pred, average='macro')
print('test:acc: %.4f  precision: %.4f  recall: %.4f  f1: %.4f')

labels11 = ['story','culture','entertainment','sports','finance',
                    'house','car','edu','tech','military',
                    'travel','world','stock','agriculture','game']

fig, ax = plt.subplots(figsize=(15, 15))
disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
disp.plot(cmap='Blues', values_format='', ax=ax)
plt.savefig('ConfusionMatrix.tif', dpi=400)

def dataParse_(content, stop_words):
    content = reserve_chinese(content)
    words = lcut(content)
    words = [word for word in words if not word in stop_words]
    return words

def getData_one(file):
    file = open(file, 'r', encoding='utf8')
    texts = file.readlines()
    file.close()
    stop_words = getStopWords()
    all_words = [] 
    word = []
    for text in texts:
        content = dataParse_(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        word.append(text)
    return all_words, word

def predict_(file):
    data_cut ,data= getData_one(file)
    t_seq = tok.texts_to_sequences(data_cut)
    t_seq_mat = pad_sequences(t_seq, maxlen=max_len)
    model = load_model('model/CNN-LSTM.h5')
    t_pre = model.predict(t_seq_mat)
    pred = np.argmax(t_pre, axis=1)
    labels11 = ['story', 'culture', 'entertainment', 'sports', 'finance',
                'house', 'car', 'edu', 'tech', 'military',
                'travel', 'world', 'stock', 'agriculture', 'game']
    pred_lable = []
    for i in pred:
        pred_lable.append(labels11[i])
    df_x = pd.DataFrame(data)
    df_y = pd.DataFrame(pred_lable)
    headerList = ['label', 'text']
    data = pd.concat([df_y, df_x], axis=1)
    data.to_csv('data.csv',header=headerList,)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    data = pd.read_csv('data.csv')
    result = data['label'].value_counts()
    result.plot(kind='bar')
    plt.show()
    # return pred_lable

def main_windows():
    # 菜单栏
    menu_def = [['Help', 'About...'], ]
    # 主窗口
    layout = [[sg.Menu(menu_def, tearoff=True)],
              [sg.Text('')],
              [sg.Text('请选择要处理的文本',font=("Helvetica", 16)),],
              [sg.Text('导入文本', size=(8, 1),font=("Helvetica", 16)), sg.Input(), sg.FileBrowse()],
              [sg.Text('')],
              [sg.Text('', size=(20, 1)), sg.Button('启动数据处理',font=("Helvetica", 16))],
              [sg.Text('')],
              [sg.Text('', size=(20, 1)), sg.Text(key="output", justification='center',font=("Helvetica", 16))],
              [sg.Text('')],
              [sg.Text('')], ]
    win1 = sg.Window('中文文本分类系统', layout)
    while True:
        ev1, vals1 = win1.Read()
        if ev1 is None:
            break
        if ev1 == '启动数据处理':
            predict_(vals1[1])
            win1['output'].update('处理完毕')
        else:
            pass

if __name__ == "__main__":
    main_windows()







# 1. ------------------------------------TextCNN----------------------------------
import os
import time
from jieba import lcut
from torchtext.vocab import vocab
from torchtext.transform import VocabTransform
from collections import Counter, OrderedDict
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.preprocessing import LabelEncoder
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.nn import functional as F
import math

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix

def metrics(pred, real):
    acc = accuracy_score(real, pred)
    precision = precision_score(real, pred, average='macro')
    recall = recall_score(real, pred, average='macro')
    f1 = f1_score(real, pred, average='macro')
    return acc, precision, f1, recall

def cost(fun):
    def use_time(*arg,**args):
        start = time.time()
        fun(*arg,**args)
        end = time.time()
        print('cost time:', end - start)
    return use_time


def safeCreateDir(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 数据处理
# """判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if (uchar >= '\u4e00' and uchar <= '\u9fa5') :
        return True
    else:
        return False

# 是中文就留下 不是就跳过
def reserve_chinese(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str += i
    return content_str

# 读取去停用词库
def getStopWords():
    file = open('./dataset/stopwords.txt', 'r',encoding='utf8')
    words = [i.strip() for i in file.readlines()]
    file.close()
    return words

# 数据清洗、分词、去停用词
def dataParse(text, stop_words):
    label_map = {'news_story': 0, 'news_culture': 1, 'news_entertainment': 2,
               'news_sports': 3, 'news_finance': 4, 'news_house': 5, 'news_car': 6,
               'news_edu': 7, 'news_tech': 8, 'news_military': 9, 'news_travel': 10,
               'news_world': 11, 'stock': 12, 'news_agriculture': 13, 'news_game': 14}
    _, _, label,content, _ = text.split('_!_')
    label = label_map[label]
    # 去掉非中文词
    content = reserve_chinese(content)
    # 结巴分词
    words = lcut(content)
    # 去停用词
    words = [i for i in words if not i in stop_words]
    return words, int(label)


def getFormatData():
    stop_words = getStopWords()
    file = open('./data/toutiao_cat_data.txt', 'r', encoding='utf8')
    texts = file.readlines()
    file.close()
    all_words = []
    all_labels = []
    for text in texts:
        content, label = dataParse(text, stop_words)
        if len(content) <= 0:
            continue
        all_words.append(content)
        all_labels.append(label)

    # 自制词汇表，将所有词汇总到一个表中
    # ws = sum([['爸爸', '妈妈', '爱', '我'], ['爸爸', '妈妈', '爱', '中国'], ['我','爱', '中国']], [])
    # ws-->['爸爸', '妈妈', '爱', '我', '爸爸', '妈妈', '爱', '中国', '我', '爱', '中国']
    # set_ws = Counter(ws)
    # set_ws-->Counter({'爸爸': 2, '妈妈': 2, '爱': 3, '我': 2, '中国': 2})
    # keys = sorted(set_ws, key=lambda x: set_ws[x], reverse=True) # 按词频排序
    # keys-->['爱', '爸爸', '妈妈', '我', '中国']
    ws = sum(all_words, [])
    set_ws = Counter(ws)
    keys = sorted(set_ws, key=lambda x: set_ws[x], reverse=True) # 按词频排序
    dict_words = dict(zip(keys, list(range(1, len(set_ws)+1))))
    ordered_dict = OrderedDict(dict_words) # OrderedDict([('爱', 1), ('爸爸', 2), ('妈妈', 3), ('我', 4), ('中国', 5)])
    
    my_vocab = vocab(ordered_dict, specials=['<UNK>', '<SEP>'])

    # 将输入的词元映射成它们在词表中的索引
    vocab_transform = VocabTransform(my_vocab)
    vector = vocab_transform(all_words)
    # 转成tensor
    vector = [torch.tensor(i) for i in vector]
    # 对tensor做padding 保证网络定长输入
    pad_seq = pad_sequence(vector, batch_first=True)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(all_labels)
    data = pad_seq.numpy()
    data = {'X': data,
            'label': labels,
            'num_words': len(my_vocab)}
    io.savemat('./dataset/data/data.mat', data)


class Data(Dataset):
    def __init__(self, mode='train'):
        data = io.loadmat('./dataset/data/data.mat')
        self.X = data['X']
        self.y = data['label']
        self.num_words = data['num_words'].item()
        train_X, val_X, train_y, val_y = train_test_split(self.X, self.y.squeeze(), test_size=0.2, random_state=42)
        val_X, test_X, val_y, test_y = train_test_split(val_X, val_y, test_size=0.5, random_state=42)
        if mode == 'train':
            self.X = train_X
            self.y = train_y
        elif mode == 'val':
            self.X = val_X
            self.y = val_y
        elif mode == 'test':
            self.X = test_X
            self.y = test_y

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]
    
class getDataLoader():
    def __init__(self, batch_size):
        train_dataset = Data(mode='train')
        val_dataset = Data(mode='val')
        test_dataset = Data(mode='test')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.num_words = train_dataset.num_words


class textCNN(nn.Module):
    def __init__(self, params):
        super(textCNN, self)._init__()
        ci = 1 # input_channel size
        kernel_num = params['kernel_num'] # output channel size
        kernel_sizes = params['kernel_sizes']
        vocab_size = params['vocab_size']
        embed_dim = params['embed_dim']
        num_classes = params['num_classes']
        dropout = params['dropout']
        self.params = params
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_sizes[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_sizes[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_sizes[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x:(batch, 1, seq_len, embed_dim)
        x = conv(x)
        # x:(batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kernel_num)
        return x
    
    def forward(self, x):
        # x:(batch, seq_len)
        x = self.embed(x)
        # x:(batch, seq_len, embed_dim)
        x = x.unsqueeze(1)
        # x:(batch, 1, seq_len, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11) # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12) # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13) # (batch, kernel_num)
        x = torch.cat([x1, x2, x3], dim=1) # # (batch, 3*kernel_num)
        x = self.dropout(x)
        x = self.fc1(x)
        logit = F.log_softmax(x, dim=1)
        return logit
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2. / n))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def plot_acc(train_acc):
        sns.set(style='darkgrid')
        plt.figure(figsize=(10, 7))
        x = list(range(len(train_acc)))
        plt.plot(x, train_acc, alpha=0.9, linewidth=2, label='train acc')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc='best')
        plt.savefig('results/acc.png', dpi=400)

def plot_loss(train_loss):
        sns.set(style='darkgrid')
        plt.figure(figsize=(10, 7))
        x = list(range(len(train_loss)))
        plt.plot(x, train_loss, alpha=0.9, linewidth=2, label='train loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig('results/loss.png', dpi=400)


class Trainer():
    def __init__(self):
        safeCreateDir('results/')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_data()
        self._init_model()

    def _init_data(self):
        data = getDataLoader(batch_size=64)
        self.traindl = data.train_loader
        self.valdl = data.val_loader
        self.testdl = data.test_loader
        self.num_words = data.num_words

    def _init_model(self):
        self.textCNN_parmas = {
            'vocab_size':self.num_words,
            'embed_dim':64,
            'num_classes':15,
            'kernel_num':16,
            'kernel_size':[3,4,5],
            'dropout':0.5
        }
        self.net = textCNN(self.textCNN_parmas)
        self.opt = Adam(self.net.parameters(), lr=0.001, weight_decay=5e-4)
        self.cri = nn.CrossEntropyLoss()

    def save_model(self):
        torch.save(self.net.state_dict(), './results/model.pth')

    def load_model(self):
        self.net.load_state_dict(torch.load('./results/model.pth'))

    def train(self, epochs):
        self.net.init_weight()
        patten = 'Epoch: %d   [===========]  cost: %.2fs;  loss: %.4f;  train acc: %.4f;  val acc:%.4f;'
        train_accs = []
        c_loss = []
        for epoch in range(epochs):
            cur_preds = np.empty(0)
            cur_labels = np.empty(0)
            cur_loss = 0
            start = time.time()
            for batch, (inputs, targets) in enumerate(self.traindl):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.net.to(self.device)

                pred = self.net(inputs)
                loss = self.cri(pred, targets)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                
                cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
                cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
                cur_loss += loss.item()

            acc, precision, f1, recall = metrics(cur_preds, cur_labels)
            val_acc, val_precision, val_f1, val_recall = self.val()
            train_accs.append(acc)
            c_loss.append(cur_loss)
            end = time.time()
            print(patten % (epoch+1,end - start,cur_loss, acc,val_acc))

        # self.save_model()
        plot_acc(train_accs)
        plot_loss(c_loss)

    @torch.no_grad()
    def val(self):
        self.net.eval()
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets) in enumerate(self.valdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.net.to(self.device)

            pred = self.net(inputs)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        self.net.train()

        return acc, precision, f1, recall

    @torch.no_grad()
    def test(self):
        self.load_model()
        self.net.eval()
        patten = 'test acc: %.4f   precision: %.4f   recall: %.4f    f1: %.4f    '
        cur_preds = np.empty(0)
        cur_labels = np.empty(0)
        for batch, (inputs, targets) in enumerate(self.testdl):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.net.to(self.device)

            pred = self.net(inputs)
            cur_preds = np.concatenate([cur_preds, pred.cpu().detach().numpy().argmax(axis=1)])
            cur_labels = np.concatenate([cur_labels, targets.cpu().numpy()])
        
        acc, precision, f1, recall = metrics(cur_preds, cur_labels)
        print(patten % (acc, precision, recall, f1))
        
        cv_conf = confusion_matrix(cur_labels, cur_preds)
        labels11 = ['story','culture','entertainment','sports','finance',
                    'house','car','edu','tech','military',
                    'travel','world','stock','agriculture','game']
        
        fig, ax = plt.subplots(figsize=(15,15))
        disp = ConfusionMatrixDisplay(confusion_matrix=cv_conf, display_labels=labels11)
        disp.plot(cmap="Blues", values_format='',ax=ax)
        plt.savefig("results/ConfusionMatrix.png", dpi=400)
        self.net.train()

if __name__ == "__main__":
    getFormatData() # 数据预处理：数据清洗和词向量
    trainer=Trainer()
    trainer.train(epochs=30) #数据训练
    trainer.test()  # 测试