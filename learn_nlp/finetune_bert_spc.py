# 句对分本分类   Sentence Pair Classification , SPC   文本蕴含
# X = [CLS]x11x12x13...x1n[SEP]x21x22x23...x2m[SEP]
# 其中x11x12x13...x1n为第一个句子的token，x21x22x23...x2m为第二个句子的token
# 其中[CLS]为分类的起始标志，[SEP]为句子的分隔标志

import numpy as np
from datasets import load_dataset, load_metric
from evaluate import load_metric
from transformers import BertTokenierFast, BertForSequenceClassification, TrainingArguments, Trainer

dataset = load_dataset('glue', 'rte')
tokenizer = BertTokenierFast.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased') # num_labels=2
metric = load_metric('glue', 'rte')

def tokenize(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

dataset = dataset.map(tokenize, batched=True)
encoded_dataset = dataset.map(lambda examples:{'labels':examples['label']}, batched=True)

columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
encoded_dataset.set_format('torch', columns=columns)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)

args = TrainingArguments(
    output_dir='./results/ftrte',
    num_train_epochs=3,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['validation'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()