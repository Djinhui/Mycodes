# 《精通Transformer》CH05



# 1. 微调BERT模型以适用于单句二元分类:情感分析
# 1.1 使用transformers的Trainer
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})


# to take entire dataset from original train 25 K AND TEST 25K
'''
imd

b_train= load_dataset('imdb', split="train")
imdb_test= load_dataset('imdb', split="test[:6250]+test[-6250:]")
imdb_val= load_dataset('imdb', split="test[6250:12500]+test[-12500:-6250]")

'''
# to take smaller portion 4K for train, 1K for test and 1K for validation
imdb_train= load_dataset('imdb', split="train[:2000]+train[-2000:]")
imdb_test= load_dataset('imdb', split="test[:500]+test[-500:]")
imdb_val= load_dataset('imdb', split="test[500:1000]+test[-1000:-500]")

print(imdb_train.shape, imdb_test.shape, imdb_val.shape), #((4000, 2), (1000, 2), (1000, 2)) # :columns:['text', 'label']

enc_train = imdb_train.map(lambda e:tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=1000)
enc_test =  imdb_test.map(lambda e:tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=1000)
enc_val =   imdb_val.map(lambda e:tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=1000)

training_args = TrainingArguments(
    output_dir='./MyIMDBModel',          # output directory
    do_train=True, 
    do_eval=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,
    logging_strategy='steps',
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='steps',
    save_strategy='epoch',
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,

)

def compute_metrics(pred):
    labels = pred.labels
    preds = pred.predictions.argmax(-1)
    precision, reacll, f1 = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
       'recall': reacll
    }


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset=enc_train,
    eval_dataset=enc_val,
    compute_metrics=compute_metrics
)

results = trainer.train()

q = [trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
pd.DataFrame(q, index=['train', 'val', 'test'])

# save the best fine-tuned model and tokenizer
model_save_path = 'MyBestIMDBModel'
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=250).to(device)
    outputs = model(inputs['input_ids'].to(device), inputs['attention_mask'].to(device))
    probs = outputs[0].softmax(1)
    return probs, probs.argmax()

model.to(device)
text = "I didn't like the movie since it bored me "
get_prediction(text)[1].item()

# 将模型封装为管道
from transformers import pipeline

model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_save_path)
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
nlp("I didn't like the movie since it bored me ")



# 1.2 使用原生pytorch训练模型
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import datasets
from datasets import load_metric, load_dataset

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')



# a demo:one step forward

model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)


texts= ["this is a good example","this is a bad example","this is a good one"]
labels= [1,0,1]
labels = torch.tensor(labels).unsqueeze(0)
encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()
print(outputs)
'''
SequenceClassifierOutput([('loss', tensor(0.6949, grad_fn=<NllLossBackward>)),
                          ('logits', tensor([[ 0.0006,  0.0778],
                                   [-0.0180,  0.0546],
                                   [ 0.0307,  0.0186]], grad_fn=<AddmmBackward>))])
'''

#Manually calculate loss
from torch.nn import functional
labels = torch.tensor([1,0,1])
outputs = model(input_ids, attention_mask=attention_mask) # 不将标签传递给模型，只生成logits
loss = functional.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()
print(outputs)
'''
SequenceClassifierOutput([('logits', tensor([[-0.1801,  0.5541],
                                   [-0.1772,  0.5738],
                                   [-0.2964,  0.5140]], grad_fn=<AddmmBackward>))])
'''

# 完整训练
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
sst2 = load_dataset('glue','sst2')
metric = load_metric('glue','sst2')

texts = sst2['train']['sentence']
labels = sst2['train']['label']
val_texts = sst2['validation']['sentence']
val_labels = sst2['validation']['label']
test_texts = sst2['test']['sentence']
test_labels = sst2['test']['label']

train_dataset = MyDataset(tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt'), labels)
val_dataset = MyDataset(tokenizer(val_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'), val_labels)
test_dataset = MyDataset(tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors='pt'), test_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(5):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

    model.eval()
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
    eval_metric = metric.compute()
    print(f'epoch :{epoch}:{eval_metric}')


# 2. 使用自定义数据集对多类别分类BERT模型进行微调:单句多分类
from torch import cuda
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 

device = 'cuda' if cuda.is_available() else 'cpu'

data = pd.read_csv('./CH5/TTC490.csv') # 土耳其语 # columns:['category', 'text']
datat = data.sample(frac=0.1)

labels=["teknoloji","ekonomi","saglik","siyaset","kultur","spor","dunya"]
NUM_LABELS= len(labels)
id2label={i:l for i,l in enumerate(labels)}
label2id={l:i for i,l in enumerate(labels)}

data['labels'] = data.category.map(lambda x: label2id[x.strip()])
SIZE= data.shape[0]

train_texts= list(data.text[:SIZE//2])
val_texts=   list(data.text[SIZE//2:(3*SIZE)//4 ])
test_texts=  list(data.text[(3*SIZE)//4:])

train_labels= list(data.labels[:SIZE//2])
val_labels=   list(data.labels[SIZE//2:(3*SIZE)//4])
test_labels=  list(data.labels[(3*SIZE)//4:])


tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased", max_length=512)
model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)
print(model)
'''
...
(classifier): Linear(in_features=768, out_features=7, bias=True)
'''

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
train_dataset = MyDataset(train_encodings, train_labels)
val_dataset = MyDataset(val_encodings, val_labels)
test_dataset = MyDataset(test_encodings, test_labels)

def compute_metrics(pred): 
    labels = pred.label_ids 
    preds = pred.predictions.argmax(-1) 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro') 
    acc = accuracy_score(labels, preds) 
    return { 
        'Accuracy': acc, 
        'F1': f1, 
        'Precision': precision, 
        'Recall': recall 
    }


training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./TTC4900Model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory                 
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch", 
    fp16=True,
    load_best_model_at_end=True
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,                 
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics= compute_metrics
)

trainer.train()

q=[trainer.evaluate(eval_dataset=data) for data in [train_dataset, val_dataset, test_dataset]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]


def predict(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs, probs.argmax(),model.config.id2label[probs.argmax().item()]


text = "Fenerbahçeli futbolcular kısa paslarla hazırlık çalışması yaptılar"
predict(text)


# saving the fine tuned model & tokenizer
model_path = "turkish-text-classification-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
nlp= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

nlp("Sinemada hangi filmler oynuyor bugün")
# [{'label': 'kultur', 'score': 0.897723913192749}]



# 3. 微调BERT模型以适用于句子对相似性回归
from transformers import DistilBertConfig, DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW, TrainingArguments, Trainer
import datasets
from datasets import load_dataset

MODEL_PATH = 'distilbert-base-uncased'
config = DistilBertConfig.from_pretrained(MODEL_PATH, num_labels=1)
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, config=config) # or 将num_labels=1传入


stsb_train= load_dataset('glue','stsb', split="train")
stsb_validation = load_dataset('glue','stsb', split="validation")
stsb_validation=stsb_validation.shuffle(seed=42)
stsb_val= datasets.Dataset.from_dict(stsb_validation[:750])
stsb_test= datasets.Dataset.from_dict(stsb_validation[750:])

pd.DataFrame(stsb_train) # columns:['idx', 'label', 'sentence1', 'sentence2']
print(stsb_train.shape, stsb_val.shape, stsb_test.shape)
# (15000, 4) (750, 4) (750, 4)

enc_train = stsb_train.map(lambda e: tokenizer(e['sentence1'], e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000)
enc_val = stsb_val.map(lambda e: tokenizer(e['sentence1'], e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000)
enc_test = stsb_test.map(lambda e: tokenizer(e['sentence1'], e['sentence2'], padding=True, truncation=True), batched=True, batch_size=1000)

pd.DataFrame(enc_train) # [attention_mask	idx	input_ids	label	sentence1	sentence2], 怎么没有token_type_ids

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./stsb-model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    # TensorBoard log directory
    logging_strategy='steps',                
    logging_dir='./logs',            
    logging_steps=50,
    # other options : no, steps
    evaluation_strategy="steps",
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True
)

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
def compute_metrics(pred):
    preds = np.squeeze(pred.predictions) 
    return {"MSE": ((preds - pred.label_ids) ** 2).mean().item(),
            "RMSE": (np.sqrt ((  (preds - pred.label_ids) ** 2).mean())).item(),
            "MAE": (np.abs(preds - pred.label_ids)).mean().item(),
            "Pearson" : pearsonr(preds,pred.label_ids)[0],
            "Spearman's Rank" : spearmanr(preds,pred.label_ids)[0]
            }


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=enc_train,
        eval_dataset=enc_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )


train_result = trainer.train()
metrics = train_result.metrics

s1,s2="A plane is taking off.",	"An air plane is taking off."
encoding = tokenizer(s1,s2, return_tensors='pt', padding=True, truncation=True, max_length=512)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)
outputs = model(input_ids, attention_mask=attention_mask)
outputs.logits.item()


q=[trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:6]


model_path = "sentence-pair-regression-model"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)