# huggingFace自然语言处理详解
'''
数据介绍本章所使用的数据集依然是ChnSentiCorp数据集，这是一个情感分类数据集，每条数据中包括一句购物评价，以及一个标识，由于本章的任务为填空任务，所以只需文本就可以了，不需要分类标识
在数据处理的过程中，会把每句话的第15个词挖掉，也就是替换成特殊符号[MASK]，并且每句话会被截断成固定的30个词的长度，神经网络的任务就是根据每句话的上下文，把第15个词预测出来
'''
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
from datasets import load_from_disk

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert_base_chinese')


# 0. =================处理数据==================
dataset = load_from_disk('./data/ChnSentiCorp')
'''
dataset:
DatasetDict({
train: Dataset({
features: ['text', 'label'],
num_rows: 9600
})
validation: Dataset({
features: ['text', 'label'],
num_rows: 0
})
test: Dataset({
features: ['text', 'label'],
num_rows: 1200
})
})

'''

def f(data):
    return tokenizer.batch_encode_plus(data['text'], max_length=30, truncation=True, padding='max_length', return_length=True)

dataset = dataset.map(f, batched=True, batch_size=1000, num_proc=4, remove_columns=['text', 'label'])
'''
dataset:
DatasetDict({
train: Dataset({
features: ['input_ids', 'token_type_ids', 'length', 'attention_mask'],
num_rows: 9600
})
validation: Dataset({
features: [],
num_rows: 0
})
test: Dataset({
features: ['input_ids', 'token_type_ids', 'length', 'attention_mask'],
num_rows: 1200
})
})
'''

def filter(data):
    return [i >= 30 for i in data['length']]

dataset = dataset.filter(filter, batched=True, batch_size=1000, num_proc=4)

def collate_fn(data):
    input_ids = [i['input_ids'] for i in data]
    attention_mask = [i['attention_mask'] for i in data]
    token_type_ids = [i['token_type_ids'] for i in data]
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

    # 掩码第15个词
    labels = input_ids[:, 15].reshape(-1,).clone()
    # input_ids[:, 15] = tokenizer.convert_tokens_to_ids('[MASK]')
    input_ids[:,15] = tokenizer.get_vocab()[tokenizer.mask_token]

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    return input_ids, attention_mask, token_type_ids, labels

loader = DataLoader(dataset['train'], batch_size=16, collate_fn=collate_fn, shuffle=True, drop_last=True)

#第8章/查看数据样例
for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break

print(tokenizer.decode(input_ids[0]))
print(tokenizer.decode(labels[0]))
print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
'''
[CLS] 位　于 友　谊 路　金 融　街，找　不 到　吃 饭 [MASK] 地　方。酒　店 刚　刚 装　修 好，有点 [SEP] 
的
(torch.Size([16, 30]),
torch.Size([16, 30]),
torch.Size([16, 30]),
tensor([4638, 6230,  511, 7313, 3221, 7315, 6820, 6858, 7564, 3211, 1690, 3315, 3300,  172, 6821, 1126], device='CUDA:0'))
'''

# 1. ====================加载预训练模型=============
pretrained_model = BertModel.from_pretrained('bert_base_chinese')
pretrained_model.to(device)
print(p.numel() for p in pretrained_model.parameters())

for p in pretrained_model.parameters():
    p.requires_grad_(False)

out = pretrained_model(input_ids, attention_mask, token_type_ids)
print(out.last_hidden_state.shape) # (16,30,768)

# 2. ====================定义下游任务模型===================
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.decoder = torch.nn.Linear(768, tokenizer.vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(tokenizer.vocab_size))
        self.decoder.bias = self.bias
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained_model(input_ids, attention_mask, token_type_ids)
        
        # 把第15个词的特征投影到全字典范围内
        out = self.dropout(out.last_hidden_state[:,15]) # (16, 21118)
        out = self.decoder(out)
        return out
    
model = Model()
model.to(device)


def train():
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=1.0)
    criterion=  torch.nn.CrossEntropyLoss()
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader)*5)

    model.train()
    for epoch in range(5):
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 50 == 0:
                out = out.argmax(dim=1)
                accuracy = (out == labels).sum().item() / len(labels)
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'epoch: {epoch}, step: {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}, lr: {lr:.6f}')

train()

def test():
    loader_test = DataLoader(dataset['test'], batch_size=16, collate_fn=collate_fn, shuffle=False, drop_last=True)
    model.eval()
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
    print(f'accuracy: {correct / total:.4f}')

test()


