# 《精通Transformer》CH06   See At finetune_bert_ner.py or ner_bert.py

# 1. 微调BERT模型NER命名实体识别
import numpy as np
import datasets 
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

conll2003 = datasets.load_dataset("conll2003")
print(conll2003['train'][0])
"""
{'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
 'id': '0',
 'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0],
 'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
 'tokens': ['EU',
  'rejects',
  'German',
  'call',
  'to',
  'boycott',
  'British',
  'lamb',
  '.']}
"""

print(conll2003['train'].features['ner_tags'])

"""
Sequence(feature=ClassLabel(num_classes=9, 
names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'], names_file=None, id=None), length=-1, id=None)
"""
label_list = conll2003["train"].features["ner_tags"].feature.names 

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
metric = datasets.load_metric("seqeval") 

def tokenize_and_align_labels(examples, label_all_tokens=True): 
    """

    https://blog.csdn.net/qq_40990057/article/details/119974316
    
    预训练模型在预训练的时候通常使用的是subword,如果我们的文本输入已经被切分成了word,那么这些word还会被我们的tokenizer继续切分
    由于标注数据通常是在word级别进行标注的,既然word还会被切分成subtokens,那么意味着还需要对标注数据进行subtokens的对齐。
    同时，由于预训练模型输入格式的要求，往往还需要加上一些特殊符号比如： [CLS] 和 a [SEP]
    当tokenizer的is_split_into_words参数设置为True时,可以分散的输入词汇，不受语句数量的限制

    tokenizer有一个word_ids方法可以实现对齐
    [IN]:print(tokenized_input.word_ids())
    [OUT]:[None, 0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, None]
    word_ids将每一个subtokens位置都对应了一个word的下标。比如第1个位置对应第0个word,然后第2、3个位置对应第1个word。特殊字符对应了NOne。
    有了这个list,就能将subtokens和words还有标注的labels对齐:
    word_ids = tokenized_input.word_ids()
    aligned_labels = [-100 if word_idx is None else example[f"ner_tags"][word_idx] for word_idx in word_ids]


    两种对齐label的方式:通过label_all_tokens = True切换
    1. 多个subtokens对齐一个word,对齐一个label
    2. 多个subtokens的第一个subtoken对齐word,对齐一个label,其他subtokens直接赋予-100.

    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) 
    labels = [] 
    for i, label in enumerate(examples["ner_tags"]): 
        word_ids = tokenized_inputs.word_ids(batch_index=i) # 一个词会被分词为多个子词，子词属于同一word_id
        previous_word_idx = None 
        label_ids = [] 
        for word_idx in word_ids: 
            if word_idx is None: 
                label_ids.append(-100) 
            elif word_idx != previous_word_idx: 
                 label_ids.append(label[word_idx]) 
            else: 
                 label_ids.append(label[word_idx] if label_all_tokens else -100) 
            previous_word_idx = word_idx 
        labels.append(label_ids) 
    tokenized_inputs["labels"] = labels 
    return tokenized_inputs 

q = tokenize_and_align_labels(conll2003['train'][4:5]) 
print(q)
"""
{'input_ids': [[101, 2762, 1005, 1055, 4387, 2000, 1996, 2647, 2586, 1005, 1055, 15651, 2837, 14121, 1062, 9328, 5804, 2056, 2006, 9317, 10390, 2323, 4965, 8351, 4168, 4017, 2013, 3032, 2060, 2084, 3725, 2127, 1996, 4045, 6040, 2001, 24509, 1012, 102]], 
'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], 
'labels': [[-100, 5, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, -100]]}

"""

for token, label in zip(tokenizer.convert_ids_to_tokens(q["input_ids"][0]),q["labels"][0]): 
    print(f"{token:_<40} {label}") 

tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched=True)

args = TrainingArguments( 
"test-ner",
evaluation_strategy = "epoch", 
learning_rate=2e-5, 
per_device_train_batch_size=16, 
per_device_eval_batch_size=16, 
num_train_epochs=3, 
weight_decay=0.01, 
) 

data_collator = DataCollatorForTokenClassification(tokenizer) 

def compute_metrics(p): 
    predictions, labels = p 
    predictions = np.argmax(predictions, axis=2) 
    true_predictions = [ 
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100] 
        for prediction, label in zip(predictions, labels) 
    ] 
    true_labels = [ 
      [label_list[l] for (p, l) in zip(prediction, label) if l != -100] 
       for prediction, label in zip(predictions, labels) 
   ] 
    results = metric.compute(predictions=true_predictions, references=true_labels) 
    return { 
   "precision": results["overall_precision"], 
   "recall": results["overall_recall"], 
   "f1": results["overall_f1"], 
  "accuracy": results["overall_accuracy"], 
  } 


trainer = Trainer(
    model, 
    args, 
   train_dataset=tokenized_datasets["train"], 
   eval_dataset=tokenized_datasets["validation"], 
   data_collator=data_collator, 
   tokenizer=tokenizer, 
   compute_metrics=compute_metrics 
) 

trainer.train() 

model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")


id2label = {
    str(i): label for i,label in enumerate(label_list)
}
label2id = {
    label: str(i) for i,label in enumerate(label_list)
}

import json
config = json.load(open("ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model/config.json","w"))


from transformers import AutoModelForTokenClassification, pipeline
model = AutoModelForTokenClassification.from_pretrained("ner_model")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "I live in Istanbul"

ner_results = nlp(example)
print(ner_results)
# [{'entity': 'B-LOC', 'score': 0.9969446, 'index': 4, 'word': 'istanbul', 'start': 10, 'end': 18}]