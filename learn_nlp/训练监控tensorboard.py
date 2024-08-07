# 《精通transformer》 CH11

# 1. ------------------------------------Tracking with TensorBoard----------------------
from transformers.trainer_utils import set_seed
set_seed(42)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
model_path = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})


from datasets import load_dataset

# to take entire dataset from original train 25 K AND TEST 25K
'''
imdb_train= load_dataset('imdb', split="train")
imdb_test= load_dataset('imdb', split="test[:6250]+test[-6250:]")
imdb_val= load_dataset('imdb', split="test[6250:12500]+test[-12500:-6250]")
'''

# to take smaller portion 4K for train, 1K for test and 1K for validation
imdb_train= load_dataset('imdb', split="train[:2000]+train[-2000:]")
imdb_test= load_dataset('imdb', split="test[:500]+test[-500:]")
imdb_val= load_dataset('imdb', split="test[500:1000]+test[-1000:-500]")

enc_train = imdb_train.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_test =  imdb_test.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_val =   imdb_val.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 

from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./MyIMDBModel', 
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
    logging_dir='./logs',            
    logging_steps=50,
    # other options : no, steps
    evaluation_strategy="steps",
    save_strategy="epoch",
    fp16=True,
    load_best_model_at_end=True,
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,
    # training and validation dataset                 
    train_dataset=enc_train,         
    eval_dataset=enc_val,            
    compute_metrics= compute_metrics
)

results=trainer.train()

q = [trainer.evaluate(eval_dataset=data) for data in [enc_train, enc_val, enc_test]]
import pandas as pd
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]

'''
in notebook
%reload_ext tensorboard
%tensorboard --logdir logs
'''

# 2. ---------------------------实时跟踪--Tracking with Wandb---------------------
import torch
from transformers.trainer_utils import set_seed
set_seed(42)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import wandb
# or
# export WANDB_API_KEY= PUT-YOUR-API-KEY-HERE
!wandb login --relogin

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
model_path= 'distilbert-base-uncased'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})

from datasets import load_dataset

# to take entire dataset from original train 25 K AND TEST 25K
'''
imdb_train= load_dataset('imdb', split="train")
imdb_test= load_dataset('imdb', split="test[:6250]+test[-6250:]")
imdb_val= load_dataset('imdb', split="test[6250:12500]+test[-12500:-6250]")

'''
# to take smaller portion 4K for train, 1K for test and 1K for validation
imdb_train= load_dataset('imdb', split="train[:2000]+train[-2000:]")
imdb_test= load_dataset('imdb', split="test[:500]+test[-500:]")
imdb_val= load_dataset('imdb', split="test[500:1000]+test[-1000:-500]")

enc_train = imdb_train.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_test =  imdb_test.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 
enc_val =   imdb_val.map(lambda e: tokenizer( e['text'], padding=True, truncation=True), batched=True, batch_size=1000) 

from transformers import TrainingArguments, Trainer
training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./MyIMDBModel', 
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
    logging_dir='./logs',            
    logging_steps=50,
    # other options : no, steps
    evaluation_strategy="steps",
    save_strategy="steps",
    fp16=True,
    load_best_model_at_end=True,
    learning_rate=5e-5,
    report_to="wandb",
    run_name="IMDB-batch-16-lr-5e-5"
    )

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

# for a fresh copy of distilBert
# model = DistilBertForSequenceClassification.from_pretrained(model_path, id2label={0:"NEG", 1:"POS"}, label2id={"NEG":0, "POS":1})

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,
    # training and validation dataset                 
    train_dataset=enc_train,         
    eval_dataset=enc_val,            
    compute_metrics= compute_metrics
)

results=trainer.train()

wandb.finish()