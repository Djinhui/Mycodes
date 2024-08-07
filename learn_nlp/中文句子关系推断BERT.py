# HuggingFace 自然语言处理详解
# 判定两个句子是否相关/无关
import random
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler
from datasets import load_from_disk

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

class MyDataset(Dataset):
    def __init__(self, split):
        super(MyDataset, self).__init__()
        dataset = load_from_disk('./data/ChnSentiCorp')[split]
        self.dataset = dataset.filter(lambda x: len(x['text']) > 40)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text = self.dataset[index]['text']
        sentence1 = text[:20]
        sentence2 = text[20:40]
        label = random.randint(0, 1)
        if label == 1:
            j = random.randint(0, len(self.dataset)-1)
            sentence2 = self.dataset[j]['text'][20:40]
        return sentence1, sentence2, label
    

def collate_fn(data):
    sents = [i[:2] for i in data]
    labels = [i[2] for i in data]

    data = tokenizer.batch_encode_plus(sents, padding='max_length', truncation=True, return_tensors='pt', max_length=45,return_length=True,
                                       add_special_tokens=True)
    
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    length = data['length'].to(device)
    labels = torch.LongTensor(labels).to(device)

    return input_ids, attention_mask, token_type_ids, labels

'''
#模拟一批数据
data = [('酒店还是非常的不错，我预定的是套间，服务', '非常好，随叫随到，结账非常快。',0),
('外观很漂亮，性价比感觉还不错，功能简', '单，适合出差携带。蓝牙摄像头都有了。',0),
('《穆斯林的葬礼》我已闻名很久，只是一直没', '怎能享受4星的服务，连空调都不能用的。', 1)]
#试算
input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
#把编码还原为句子
print(token.decode(input_ids[0]))
input_ids.shape, attention_mask.shape, token_type_ids.shape, labels


[CLS] 酒店还是非常的不错，我预定的是套间，服务 [SEP] 非常好，随叫随到，结账非常快。 [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

(torch.Size([3, 45]),
torch.Size([3, 45]),
torch.Size([3, 45]),
tensor([0, 0, 1], device='CUDA:0'))
'''

dataset = MyDataset('train')
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)

pretrained_model = BertModel.from_pretrained('bert-base-chinese').to(device)
for param in pretrained_model.parameters():
    param.requires_grad = False


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained_model(input_ids, attention_mask, token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0, :])
        out = out.Softmax(dim=1)
        return out
    
model = Model()
model.to(device)

def train():
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader)*10)
    model.train()

    for epoch in range(10):
        for i, batch in enumerate(loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            out = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if i % 100 == 0:
                # get accuracy
                out = out.argmax(dim=1)
                acc = (out == labels).sum().item() / len(labels)
                # get optimizer learning rate
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'epoch: {epoch}, step: {i}, loss: {loss.item()}, acc: {acc}, lr: {lr}')
train()


def test():
    loader_test = DataLoader(MyDataset('test'), batch_size=32, collate_fn=collate_fn, shuffle=True, drop_last=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader_test):
            input_ids, attention_mask, token_type_ids, labels = batch
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            correct += (out == labels).sum().item()
            total += len(labels)
    print(f'accuracy: {correct / total}')


test()