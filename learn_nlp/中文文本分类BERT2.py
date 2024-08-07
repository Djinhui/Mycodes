# huggingface自然语言处理详解
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from transformers import BertTokenizer,BertForSequenceClassification,AutoModelForSequenceClassification
from datasets import load_from_disk
from transformers import AdamW,get_scheduler


device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = load_from_disk('./data/ChnSentiCorp')
dataset
'''
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

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]

    data = tokenizer.batch_encode_plus(sents, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt', return_length=True)
    # input_ids = data['input_ids'].to(device)
    # attention_mask = data['attention_mask'].to(device)
    # token_type_ids = data['token_type_ids'].to(device)
    for k, v in data.items():
        data[k] = v.to(device)

    data['labels'] = torch.LongTensor(labels).to(device)
    return data

loader = DataLoader(dataset=dataset['train'],batch_size=16,collate_fn=collate_fn,
                    shuffle=True,drop_last=True)

model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=2)
model.to(device)
print(sum(p.numel() for p in model.parameters()))

print(model)
'''
BertForSequenceClassification(
(bert): BertModel(
...
)
(DropOut): DropOut(p=0.1, inplace=False)
(classifier): Linear(in_features=768, out_features=2, bias=True)
)
'''

def train():
    optimizer = AdamW(model.parameters(), lr=5e-4)
    scheduler = get_scheduler(name='linear', num_warmup_steps=0,
                              num_training_steps=len(loader),optimizer=optimizer)
    
    model.train()
    for t, data in enumerate(loader):
        out = model(**data)  # out['loss'], out['logits']
        out['loss'].backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        model.zero_grad()

        if t % 10 == 0:
            out = out['logits'].argmax(dim=1)
            acc = (out == data['labels']).sum().item() / len(data['labels'])
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(t, out['loss'].item(), lr, acc)

train()

def test():
    loader_test = DataLoader(dataset['test'],batch_size=32, collate_fn=collate_fn, shuffle=False, drop_last=True)

    
    model.eval()
    correct, total = 0, 0
    for t, data in enumerate(loader_test):
        with torch.no_grad():
            out = model(**data)  # out['loss'], out['logits']

        out = out['logits'].argmax(dim=1)
        correct = (out == data['labels']).sum().item()
        total += len(data['labels'])
        print(correct/total)

test