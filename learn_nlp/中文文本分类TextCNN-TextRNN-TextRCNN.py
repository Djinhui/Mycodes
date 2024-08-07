# ref:《自然语言处理应用与实战》Chp7

import os
import pickle as pkl
import torch.backends
import torch.backends.cudnn
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import torch.autograd as autograd
import time
from tensorboardX import SummaryWriter

MAX_VOCAB_SIZE = 10000 # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>' # 未知词、padding符号


class Config:
    def __init__(self, embedding):
        self.model_name = 'TextCNN'
        self.train_path = './data/train.txt'
        self.dev_path = './data/dev.txt'
        self.test_path = './data/test.txt'
        self.predict_path = './data/predict.txt'
        self.class_list = [x.strip() for x in open('./data/class.txt', encoding='utf-8').readlines()]
        self.vocab_path = './data/vocab.pkl'
        self.save_path = './saved_dict/' + self.model_name + '.cpkt'
        self.log_path = './logs/'+self.model_name
        self.embedding_pretrained = torch.tensor(np.load(embedding)['embeddings'].astype('float32')) if embedding!='random' else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 1000 # early_stopping
        self.num_classes = len(self.class_list)
        self.n_vocab = None
        self.num_epochs = 5
        self.batch_size = 32
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 300
        self.filter_sizes = (2,3,4) # 卷积核尺寸
        self.num_filters = 256
        self.hidden_size = 256
        self.num_layers = 2

class TextCNN(nn.Module):
    def __init__(self, config:Config):
        super(TextCNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab-1)

        self.convs = nn.ModuleList([
            nn.Conv2d(1,config.num_filters, (k, config.embed)) for k in config.filter_sizes
        ]) #  use Conv1d 
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters*len(config.filter_sizes), config.num_classes)

    def conv_add_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_add_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
    
class TextRNN(nn.Module):
    def __init__(self, config:Config):
        super(TextRNN, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False
            )
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed,padding_idx=config.n_vocab-1)

        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirection=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size*2, config.num_classes)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])
        return out


class TextRCNN(nn.Module):
    def __init__(self, config:Config):
        super(TextRCNN,self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False
            )
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed,padding_idx=config.n_vocab-1)

        self.lstm = nn.LSTM(config.embed, config.hidden_size,
                            config.num_layers, bidirectional=True,dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size*2 + config.embed, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out,_ = self.lstm(embed)
        out = torch.cat((embed, out),2)
        out = F.relu(out)
        out = out.permute(0,2,1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out




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
    return vocab_dict # word2id


def build_dataset(config, ues_word):
    if ues_word:
        tokenizer = lambda x: x.split(' ') # word-level
    else:
        tokenizer = lambda x:[y for y in x] # char-level

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
        print(f'Vocab size: {len(vocab)}')

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
    return train, dev, test


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

import time
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif

train_dir = './THUCNews/data/train.txt'
vocab_dir = './THUCNews/data/vocab.pkl'
pretrain_dir = './THUCNews/data/sgns.sogou.char'
emb_dim = 300
filename_trimmed_dir = './THUCNews/data/embedding_SougouNews'
if os.path.exists(vocab_dir):
    word_to_id = pkl.load(open(vocab_dir, 'rb'))
else:
    tokenizer = lambda x:[y for y in x] # 以字为单位构建词表
    word_to_id = build_vocab(train_dir, tokenizer, min_freq=1, max_size=MAX_VOCAB_SIZE)
    pkl.dump(word_to_id, open(vocab_dir,'wb'))

embeddings = np.random.rand(len(word_to_id), emb_dim)
f = open(pretrain_dir, 'r', encoding='utf-8')
for i, line in enumerate(f.readlines()):
    line = line.strip().split(' ')
    if line[0] in word_to_id:
        idx = word_to_id[line[0]]
        emb = [float(x) for x in line[1:301]]
        embeddings[idx] = np.asarray(emb,dtype='float32')

f.close()
np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)


def init_netword(model:nn.Module, method='xavier', exclude='embedding', seed=123):
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
                nn.init.constant_(w, 0)
            else:
                pass


def train(config:Config, model:nn.Module, train_iter:DatasetIterater, 
          dev_iter:DatasetIterater, test_iter:DatasetIterater):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False
    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M', time.localtime()))
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

                time_dif = get_time_dif(start_time)
                print(f'{time_dif}')
                msg = 'Iter:{0:>6} Train Loss:{1:>5.2} Train Acc:{2:>6.2%} Val Loss:{3:>5.2} Val Acc:{4:>6.2%}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc))
                writer.add_scaler("loss/train",loss.item(), total_batch)
                writer.add_scaler("loss/dev", dev_loss, total_batch)
                
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                print('Early Stopping')
                flag = True
                break
        if flag:
            break

    writer.close()
    test(config, model, test_iter)


def evaluate(config:Config, model:nn.Module, data_iter:DatasetIterater, test=False):
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

def test(config:Config, model:nn.Module, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    print(test_acc, test_loss)
    print(test_report)
    print(test_confusion)


def load_dataset(text, vocab, config, pad_size=32):
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

def final_predict(config:Config, model:nn.Module, data_iter:DatasetIterater):
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



from importlib import import_module
import argparse
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model:TextCNN TextRNN TextRCNN')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pretrained')
parser.add_argument('--word', default=False, type=bool, help='True for word False for char')
args = parser.parse_args()


dataset = 'THUCNews'
embedding = 'embedding_SougouNews.npz'
if args.embedding == 'random':
    embedding = 'random'


model_name = args.__module__
x = import_module('models.'+model_name)
config = x.Config(dataset, embedding)
np.random.seed(11)
torch.manual_seed(11)
torch.cuda.manual_seed_all(11)
torch.backends.cudnn.deterministic = True

start_time = time.time()
vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
train_iter = build_iterator(train_data, config, False)
dev_iter = build_iterator(dev_data, config, False)
test_iter = build_iterator(test_data, config, False)
time_dif = get_time_dif(start_time)

config.n_vocab = len(vocab)
model = x.Model(config).to(config.device)
train(config, model, train_iter, dev_iter, test_iter)


def main(text):
    dataset = 'THUCNews'
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'


    model_name = args.__module__
    x = import_module('models.'+model_name)
    config = x.Config(dataset, embedding)
    vocab = pkl.load(open(config.vocab_path, 'rb'))
    content = load_dataset(text, vocab, 64)
    predict = True
    predict_iter = build_iterator(content, config, predict)
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    result = final_predict(config, model, predict_iter)
    for i, j in enumerate(result):
        print(f'text:{i}:label:{j}')


text = ['国考报名序号查询后务必牢记。报名参加国考的考生：如您已经通过资格审查，请按时缴费。']
main(text)

