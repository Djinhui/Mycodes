import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from TorchCRF import CRF
import torch.optim as optim
from sklearn.metrics import classification_report

def load_data(path:str='renmindata.pkl'):
    with open(path, 'r', encoding='utf-8') as f:
        word2id = pickle.load(f)
        id2word = pickle.load(f)
        tag2id = pickle.load(f)
        id2tag = pickle.load(f)
        x_train = pickle.load(f)
        y_train = pickle.load(f)
        x_test = pickle.load(f)
        y_test = pickle.load(f)
        x_valid = pickle.load(f)
        y_valid = pickle.load(f)
        return word2id, tag2id, x_train, x_test, x_valid, y_train, y_test, y_valid, id2tag
    
word2id, tag2id, x_train, x_test, x_valid, y_train, y_test, y_valid, id2tag = load_data()


class NERDataset(Dataset):
    def __init__(self, X, Y, *args, **kwargs):
        self.data = [{'x':X[i], 'y':Y[i]} for i in range(X.shape[0])]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
class Config:
    embedding_dim = 100
    hidden_dim = 200
    vocab_size = len(word2id)
    num_tags = len(tag2id)
    dropout = 0.2
    lr = 0.001
    weight_decay = 1e-5


class NERLSTM_CRF(nn.Module):
    def __init__(self, config:Config):
        super(NERLSTM_CRF, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.num_tags = config.num_tags
        self.embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim//2, num_layers=1, bidirection=True,batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags)

    def forward(self, x, mask):
        """
        crf.viterbi_decode(h, mask)
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
             in mini batch (batch_size, batch_size)
        :return: labels of each sequence in mini batch
        """
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats)) # (batch_size, seq_len, num_labels)

        outputs = self.crf.viterbi_decode(emissions, mask)
        return outputs # LIST[LIST[int]]
    
    def log_likelihood(self, x, labels, mask):
        
        """
        crf.forward(h, labels, mask)
        :param h: hidden matrix (batch_size, seq_len, num_labels)
        :param labels: answer labels of each sequence
               in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
             in mini batch (batch_size, seq_len)
        :return: The log-likelihood (batch_size)
        """
        embeddings = self.embeds(x)
        feats, hidden = self.lstm(embeddings)
        emissions = self.linear(self.dropout(feats))
        loss = -self.crf.forward(emissions, labels, mask)

        return torch.sum(loss)
    

def parse_tags(text, path):
    tags = [id2tag[idx] for idx in path]
    begin = 0
    res = []
    for idx, tag in enumerate(tags):
        # 将连续的同类型的字连接起来
        if tag.startswith("B"):
            begin = idx
        elif tag.startswith('E'):
            end = idx
            word = text[begin:end+1]
            label = tag[2:] # E_xxx
            res.append((word, label))
        elif tag == 'O':
            res.append((text[idx],tag))

    return res




def utils_to_train():
    device = torch.device('cpu')
    max_epoch = 1
    batch_szie = 32
    num_workers = 4
    train_dataset = NERDataset(x_train, y_train)
    valid_dataset = NERDataset(x_valid, y_valid)
    test_dateset = NERDataset(x_test, y_test)

    train_dl = DataLoader(train_dataset,batch_size=batch_szie,shuffle=True,num_workers=num_workers)
    vaild_dl = DataLoader(valid_dataset,batch_size=batch_szie, shuffle=True,num_workers=num_workers)
    test_dl = DataLoader(test_dateset, batch_size=batch_szie, shuffle=False,num_workers=num_workers)

    config = Config()
    model = NERLSTM_CRF(config=config).to(device)
    optimizer = optim.Adam(model.parameters(),lr=config.lr,
                           weight_decay=config.weight_decay)
    return max_epoch, device, train_dl, vaild_dl, test_dl, optimizer, model


max_epoch, device, train_dl, vaild_dl, test_dl, optimizer, model = utils_to_train()

class ChineseNER:
    def train(self):
        for epoch in range(max_epoch):
            model.train()
            for index, batch in enumerate(train_dl):
                optimizer.zero_grad()
                x = batch['x'].to(device)
                mask = (x > 0).to(device)
                y = batch[y].to(device)
                loss = model.log_likelihood(x,y,mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),max_norm=10)
                optimizer.step()

                if index % 200 == 0:
                    print('epoch:%5d -- loss:%f' %(epoch, loss.item()))

            aver_loss = 0
            preds, labels = [], []
            for index, batch in enumerate(vaild_dl):
                model.eval()
                val_x, val_y = batch['x'].to(device), batch['y'].to(device)
                val_mask = (val_x > 0).to(device)
                predict = model(val_x, val_mask)
                loss = model.log_likelihood(val_x, val_y, val_mask)
                aver_loss += loss.item()
                leng = []
                for i in val_y.cpu():
                    tmp = []
                    for j in i:
                        if j.item() > 0:
                            tmp.append(j.item())
                    leng.append(tmp)
                for index, i in enumerate(predict):
                    preds += i[:len(leng(index))]
                for index, i in enumerate(val_y.tolist()):
                    labels += i[:len(leng[index])]

            aver_loss /= (len(vaild_dl) * 64)
            report = classification_report(labels, preds)
            torch.save(model.state_dict(), 'params.pkl')


    def predict(self, input_str=""): # 输入为单句，输出为对应的单词和标签
        model.load_state_dict(torch.load('params.pkl'))
        model.eval()
        if not input_str:
            input_str = input('请输入文本:')
        input_vec = []
        for char in input_vec:
            if char not in word2id:
                input_vec.append(word2id['[unknown]'])
            else:
                input_vec.append(word2id[char])

        sentences = torch.tensor(input_vec).view(1, -1).to(device)
        mask = sentences > 0
        paths = model(sentences, mask)
        res = parse_tags(input_str, paths[0])
        return res
    
    def test(self, test_dl):
        model.load_state_dict(torch.load('params.pkl'))
        model.eval()

        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(test_dl):
            val_x, val_y = batch['x'].to(device), batch['y'].to(device)
            val_mask = (val_x > 0).to(device)
            predict = model(val_x, val_mask)
            loss = model.log_likelihood(val_x, val_y, val_mask)
            aver_loss += loss.item()
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)
            for index, i in enumerate(predict):
                preds += i[:len(leng(index))]
            for index, i in enumerate(val_y.tolist()):
                labels += i[:len(leng[index])]

        aver_loss /= (len(vaild_dl) )
        print(classification_report(labels, preds))
