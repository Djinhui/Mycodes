# huggingFace自然语言处理详解
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from datasets import load_dataset, load_from_disk


device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')

# 1. ==========数据处理
class MyDataset(Dataset):
    def __init__(self, split):
        # dataset = load_dataset('people_daily_ner', split=split)
        dataset = load_from_disk('./people_daily_ner')[split]
        self.dataset = dataset
        # dataset.features['ner_tags'].feature.num_classes # 7
        # dataset.features['ner_tags'].feature.names # ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        tokens = self.dataset[index]['tokens']
        labels = self.dataset[index]['ner_tags']
        return tokens, labels
    
dataset = MyDataset('train')
tokens, labels = dataset[0]
print(tokens)
print(labels)
'''
['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。']
[0, 0, 0, 0, 0, 0, 0, 5, 6, 0, 5, 6, 0, 0, 0, 0, 0, 0]
'''

def collate_fn(data):
    tokens = [i[0] for i in data]
    labels = [i[1] for i in data]
    tokens = tokenizer(tokens, truncation=True, padding=True, return_tensors='pt',max_length=512,
                       is_split_into_words=True) # 原始文本已经分词
    lens = tokens['input_ids'].shape[1]
    for i in range(len(labels)):
        labels[i] = [7] + labels[i]
        labels[i] += [7] * lens
        labels[i] = labels[i][:lens]
    
    for k, v in tokens.items():
        tokens[k] = v.to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    return tokens, labels

loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)



# 2. =============定义预训练模型及下游模型
pretrained = AutoModel.from_pretrained('hfl/rbt3')
pretrained.to(device)

'''
两段式训练是一种训练技巧，指先单独对下游任务模型进行一定的训练，待下游任务模型掌握了一定的知识以后，
再连同预训练模型和下游任务模型一起进行训练的模式
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.tuning = False
        self.pretrained = None
        self.rnn = nn.GRU(input_size=768, hidden_size=768, batch_first=True)
        self.fc = nn.Linear(768, 8)

    def forward(self, inputs):
        if self.tuning:
            out = self.pretrained(**inputs).last_hidden_state
        else:
            with torch.no_grad():
                out = pretrained(**inputs).last_hidden_state

        out, _ = self.rnn(out)
        out = self.fc(out).Softmax(dim=2)
        return out
    
    def fine_tuning(self, tuning):
        self.tuning = tuning
        if tuning:
            for i in pretrained.parameters():
                i.requires_grad = True
            pretrained.train()
            self.pretrained = pretrained
        else:
            for i in pretrained.parameters():
                i.requires_grad = False
            pretrained.eval()
            self.pretrained = None

model = Model()
model.to(device)

def reshape_and_remove_pad(outs, labels, attention_mask):
    outs = outs.reshape(-1, 8) # [b*lens, 8]
    labels = labels.reshape(-1) # [b*lens,]
    select = attention_mask.reshape(-1) == 1 # [b*lens]
    outs = outs[select]
    labels = labels[select]
    return outs, labels


def get_correct_and_total_count(labels, outs):
    outs = outs.argmax(dim=1) # (b*lens, 8) -->(b*lens)
    correct = (outs==labels).sum().items()
    total = len(labels)
    select = labels != 0  # 计算除了0以外元素的正确率，因为0(Other)太多，正确率会虚高
    outs = outs[select]
    labels = labels[select]
    correct_content = (outs==labels).sum().item()
    total_content = len(labels)
    return correct, total, correct_content, total_content



def train(epochs):
    lr = 1e-5 if model.tuning else 1e-5
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(name='linear', num_warmup_steps=0, num_training_steps=len(loader)*epochs, optimizer=optimizer)

    model.train()
    for epoch in range(epochs):
        for step, (inputs, labels) in enumerate(loader):
            outs = model(inputs) # (b, lens)-->(b, lens, 8)
            outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])

            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % 100 == 0:
                counts = get_correct_and_total_count(labels=labels, outs=outs)
                acc = counts[0] / counts[1]
                acc_content = counts[2] / counts[3]
                lr = optimizer.state_dict()['param_grouped'][0]['lr']
                print(epoch, step, loss.item(), lr, acc, acc_content)

    torch.save(model, 'ner_model.model')


# 第一阶段，固定预训练模型，训练下游模型
model.fine_tuning(False)
print(sum(p.numel() for p in model.parameters()))
train(10)

# 第二阶段 同时训练预训练模型和下游模型
model.fine_tuning(True)
print(sum(p.numel() for p in model.parameters()))
train(5)


def test():
    model_load = torch.load('ner_model.model')
    model_load.eval()
    model_load.to(device)

    loader_test = DataLoader(MyDataset('validation'),
                             batch_size=128, collate_fn=collate_fn,
                             shuffle=False, drop_last=True)
    correct = 0
    total = 0
    correct_content = 0
    total_content = 0
    for step, (inputs, labels ) in enumerate(load_dataset):
        with torch.no_grad():
            outs = model_load(inputs)
        outs, labels = reshape_and_remove_pad(outs, labels, inputs['attention_mask'])

        counts = get_correct_and_total_count(labels, outs)
        correct += counts[0]
        total += counts[1]
        correct_content = counts[2]
        total_content += counts[3]

    print(correct/total, correct_content/total_content)

test()


def predict():
    model_load = torch.load('ner_model.model')
    model_load.eval()
    model_load.to(device)
    loader_test = DataLoader(MyDataset('validation'),
                             batch_size=32, collate_fn=collate_fn,
                             shuffle=False, drop_last=True)
    for i, (inputs, labels) in enumerate(loader_test):
        break

    with torch.no_grad():
        outs = model_load(inputs).argmax(dim=2)

    for i in range(32):
        select = inputs['attention_mask'][i] == 1
        input_id = inputs['inputs_id'][i, select]
        out = outs[i, select]
        label = labels[i, select]
        # 输出原句子
        print(tokenizer.decode(input_id).replace(' ', ''))
        for tag in [label, out]:
            s = ''
            for j in range(len(tag)):
                if tag[j] == 0:
                    s += '.'
                    continue
                s += tokenizer.decode(input_id[j])
                s += str(tag[j].item())
            print(s)
        print('-----------------------------------')

