# X=[CLS]x1x2...xn[SEP]

import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

dataset = load_dataset("conll2003")
metric = load_metric("seqeval")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
label_list = dataset["train"].features[f"ner_tags"].feature.names
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(label_list))
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) #  is_split_into_words=True

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # 将特殊符号的标签设置为-100，以便在计算损失函数时自动忽略
            elif word_idx!= previous_word_idx: # 把标签设置到每个词的第一个token上
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx]) # 对于每个词的其他token也设置为当前标签
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, load_from_cache_file=False)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l!= -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l!= -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()