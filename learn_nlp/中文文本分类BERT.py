# HuggingFace自然语言处理详解
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import BertTokenizer, BertModel, AdamW, get_scheduler


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. ===========加载预训练模型========================

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
pretrained = BertModel.from_pretrained('bert-base-chinese')
print(sum(p.numel) for p in pretrained.parameters()) # 统计参数量

for param in pretrained.parameters():
    param.requires_grad = False

class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk('./data/ChnSentiCorp')[split]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        text = self.dataset[index]['text']
        label = self.dataset[index]['label']

        return text, label
    
dataset = Dataset('train')
print(len(dataset), dataset[20])
# (9600, ('非常不错，服务很好，位于市中心区，交通方便，不过价格也高！', 1))

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = tokenizer.batch_encode_plus(sents, padding='max_length', truncation=True, max_length=500, return_tensors='pt', return_length=True)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels

'''
#模拟一批数据
data = [
('你站在桥上看风景', 1),
('看风景的人在楼上看你', 0),
('明月装饰了你的窗子', 1),
('你装饰了别人的梦', 0),
]
#试算
input_ids, attention_mask, token_type_ids, labels = collate_fn(data)
input_ids.shape, attention_mask.shape, token_type_ids.shape, labels

(torch.Size([4, 500]),
torch.Size([4, 500]),
torch.Size([4, 500]),
tensor([1, 0, 1, 0], device='CUDA:0'))
'''

loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
len(loader) # 9600 / 16 = 600

for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
    break

'''
input_ids.shape, attention_mask.shape, token_type_ids.shape, labels
(torch.Size([16, 500]),
torch.Size([16, 500]),
torch.Size([16, 500]),
tensor([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1], device='CUDA:0'))
'''

pretrained.to(device)
out = pretrained(input_ids, attention_mask, token_type_ids)
print(out.last_hidden_state.shape) # (16, 500, 768)

# 2. =============定义下游任务模型================
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids, attention_mask, token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0, :]) # 取CLS

        out = torch.softmax(out, dim=1)
        return out
    
model = Model().to(device)
out = model(input_ids, attention_mask, token_type_ids)
print(out.shape) # (16, 2)

# 3. ==================训练==============
def train():
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = get_scheduler('linear', optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(loader))

    model.train()
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader):
        out = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(out, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            out = out.argmax(dim=1)
            accuracy = (out == labels).sum().item() / len(labels)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'loss: {loss.item()}, accuracy: {accuracy}, lr: {lr}')

train()

# 4. ==================测试==================
def test():
    loader_test = DataLoader(Dataset('test'), batch_size=16, collate_fn=collate_fn, shuffle=False, drop_last=True)

    model.eval()
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(loader_test):
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print(f'accuracy: {correct / total}')

test()

