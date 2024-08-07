import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import os
import pickle
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn import metrics


class Config:
    def __init__(self, dataPath, embeding):
        self.embeding = embeding
        self.dataPath = dataPath
        self.model_name = 'Transformer'
        self.train_path = dataPath + '/data/train.txt'
        self.dev_path = dataPath + '/data/dev.txt'
        self.test_path = dataPath + '/data/test.txt'
        self.vocab_path = dataPath + '/data/vocab.pkl' # 词表
        self.class_list = [x.strip() for x in open(dataPath + '/data/class.txt', encoding='utf-8').readlines()]
        self.save_path = dataPath + '/saved_dict/' + self.model_name + '.ckpt'
        self.log_path = dataPath + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0 # 词表大小，在运行时赋值
        self.num_epochs = 30
        self.batch_size = 128
        self.pad_size = 32
        self.dropout = 0.5
        self.learning_rate = 1e-3
        self.embed = 300
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, pad_size, dropout=0.1, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        self.pe = torch.tensor([[pos / (10000.0**(i//2*2.0/d_model)) for i in range(d_model)] for pos in range(pad_size)])
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        
    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        return self.dropout(out)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim_model, hidden)
        self.linear2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.linear2(F.relu(self.linear1(x)))
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        return out
    
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class MultiheadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_head = num_head
        self.head_dim = dim_model // num_head
        assert self.dim_model % self.num_head == 0, "dim_model must be divisible by num_head"

        self.fc_q = nn.Linear(dim_model, dim_model)
        self.fc_k = nn.Linear(dim_model, dim_model)
        self.fc_v = nn.Linear(dim_model, dim_model)

        self.attention = ScaleDotProductAttention()
        self.fc = nn.Linear(self.dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        q = q.view(batch_size*self.num_head, -1,  self.head_dim)
        k = k.view(batch_size*self.num_head, -1,  self.head_dim)
        v = v.view(batch_size*self.num_head, -1,  self.head_dim)
        scale_factor = 1 / math.sqrt(self.head_dim)
        out = self.attention(q, k, v, scale_factor)
        out = out.view(batch_size, -1, self.dim_model)
        out = self.fc(out)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiheadAttention(dim_model, num_head, dropout)
        self.feedforward = PositionwiseFeedForward(dim_model, hidden, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.feedforward(x)
        return x


class Model(nn.Module):
    def __init__(self, config:Config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.position_embedding = PositionalEncoding(config.embed, config.pad_size,config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(config.num_encoder)])
        self.fc = nn.Linear(config.dim_model*config.pad_size, config.num_classes)


    def forward(self, x):
        x = self.embedding(x)
        x = self.position_embedding(x)
        for encoder in self.encoders:
            x = encoder(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
    
def init_model(model:nn.Module, method='xavier', exclude='embedding', seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.0)
            else:
                pass
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            content = line.split('\t')[0]
            for word in tokenizer(content):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1

    vocab_list = sorted([_ for _ in vocab_dict.items() if _[1] >= min_freq], key=lambda x:x[1], reverse=True)[:max_size]
    vocab_dict = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dict.update({UNK: len(vocab_dict), PAD: len(vocab_dict) + 1})
    return vocab_dict


def build_dataset(config, ues_word):
    if ues_word: # 以空格隔开，按词构建向量
        tokenizer = lambda x:x.split(' ')
    else:
        tokenizer = lambda x:[char for char in x] # 按照单字的词构建词向量

    vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content, label = line.split('\t')
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD]*(pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                for word in token: # word---> id
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))

        return contents
    
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size,device) -> None:
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device) # pad前的长度，超过pad_size截断为pad_size
        return (x, seq_len), y
    
    def __next__(self):
        if self.residue and self.index == self.n_batches: # ?
            batches = self.batches[self.index*self.batch_size:len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index*self.batch_size:(self.index+1)*self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config, predict):
    if predict:
        config.batch_size = 1
    iterator = DatasetIterater(dataset, config.batch_size, config.device)
    return iterator

def train(config:Config, model:Model, train_iter:DatasetIterater, dev_iter:DatasetIterater):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains,  labels) in enumerate(train_iter):
            outputs = model(trains)

            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                labels = labels.data.cpu()
                predict = torch.max(outputs.data,1)[1].cpu()
                train_acc = metrics.accuracy_score(labels, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                msg = 'Iter:{0:>6} Train Loss:{1:>5.2} Train Acc:{2:>6.2%} Val Loss:{3:>5.2} Val Acc:{4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print('Early Stopping')
                flag = True
                break
        if flag:
            break


def evaluate(config:Config, model:Model, data_iter:DatasetIterater, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total/len(data_iter)


tokenizer = lambda x:[y for y in x] # char-level

def test(config:Config, model:Model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    print(test_acc, test_loss)
    print(test_report)
    print(test_confusion)


def load_dataset(text, vocab, pad_size=32):
        contents = []
        
        for line in text:
            line = line.strip()
            if not line:
                continue

            words_line = []
            token = tokenizer(line)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([PAD]*(pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size

            for word in token: # word---> id
                words_line.append(vocab.get(word, vocab.get(UNK)))
            contents.append((words_line, int(0), seq_len))

        return contents


def match_label(pred, config:Config):
    label_list = config.class_list
    return label_list[pred]

def final_predict(config:Config, model:Model, data_iter:DatasetIterater):
    map_location = lambda storage, loc:storage
    model.load_state_dict(torch.load(config.save_path, map_location=map_location))
    model.eval()
    predict_all = np.array([])
    with torch.no_grad():
        for texts, _ in data_iter:
            outputs = model(texts)
            pred = torch.max(outputs, 1)[1].cpu().numpy()
            pred_label = [match_label(i, config) for i in pred]
            predict_all = np.append(predict_all, pred_label)
    return predict_all


class TransformerPredict:
    def predict(self, text):
        content = load_dataset(text, vocab)
        predict_iter = build_iterator(content, config, predict=True)
        config.n_vocab = len(vocab)
        result = final_predict(config, model, predict_iter)
        for i,j in enumerate(result):
            print(f'text:{text[i]}, label:{j}')

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True


MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'
dataPath = 'THUCNews'
embedding = 'random'
model_name = 'Transformer'
config = Config(dataPath, embedding)

vocab, train_dt, dev_dt, test_dt = build_dataset(config, False)
train_dl = build_iterator(train_dt, config, False)
test_dl = build_iterator(test_dt, config, False)
dev_dl = build_iterator(dev_dt, config, False)
config.n_vocab = len(vocab)

model = Model(config)
init_model(model)

train(config, model, train_dl, dev_dl, test_dl)
tp = TransformerPredict()
test = ['北京举办奥运会']
tp.predict(test) # '运动

