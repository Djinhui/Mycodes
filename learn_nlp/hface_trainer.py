from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, load_metric
from evaluate import load
import torch
import numpy as np
from transformers.trainer_utils import EvalPrediction
from transformers.data.data_collator import DataCollatorWithPadding


tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
dataset = load_from_disk('./ChnSentiCorp')
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(2000))
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(100))

print(dataset)
'''
DatasetDict({
train: Dataset({
features: ['text', 'label'],
num_rows: 2000
})
validation: Dataset({
features: ['text', 'label'],
num_rows: 0
})
test: Dataset({
features: ['text', 'label'],
num_rows: 100
})
})
'''

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize, batched=True,batch_size=1000,num_proc=4)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
print(tokenized_datasets)
'''
DatasetDict({
train: Dataset({
features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
num_rows: 2000
})
validation: Dataset({
features: ['text', 'label'],
num_rows: 0
})
test: Dataset({
features: ['label', 'input_ids', 'token_type_ids', 'attention_mask'],
num_rows: 100
})
})
'''
def filter(examples):
    return [len(ex) <= 512 for ex in examples["input_ids"]]

tokenized_datasets = tokenized_datasets.filter(filter,batched=True,batch_size=1000,num_proc=4)


model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3", num_labels=2)
sum([p.nelements() for p in model.parameters()])/10000  # 3800w


#模拟一批数据
data = {
'input_ids': torch.ones(4, 10, dtype=torch.long),
'token_type_ids': torch.ones(4, 10, dtype=torch.long),
'attention_mask': torch.ones(4, 10, dtype=torch.long),
'labels': torch.ones(4, dtype=torch.long)
}
#模型试算
out = model(**data)
print(out['loss'], out['logits'].shape)

'''
(tensor(0.3597, grad_fn=<NllLossBackward0>), torch.Size([4, 2]))
'''

metric = load_metric('accuracy')

def compute_metrics(p: EvalPrediction):
    logits, labels = p
    logits = np.argmax(logits, axis=1)
    return metric.compute(predictions=logits, references=labels)

eval_pred = EvalPrediction(logits=np.array([[0.1, 0.9], 
                                            [0.9, 0.1], 
                                            [0.8, 0.2], 
                                            [0.35, 0.65]]), 
                                            label_ids=np.array([1, 0, 0, 1]))

compute_metrics(eval_pred)

args = TrainingArguments(output_dir="test_trainer",
                           evaluation_strategy="epoch",
                           per_device_train_batch_size=16,
                           per_device_eval_batch_size=16,
                           evaluation_strategy="steps",
                           eval_steps=30,
                           save_steps=30,
                           save_strategy='steps',
                           num_train_epochs=1,
                           learning_rate=0.01,
                           weight_decay=1e-2)


trainer = Trainer(model=model,
                args=args, 
                train_dataset=dataset['train'],
                eval_dataset=dataset['test'],
                compute_metrics=compute_metrics,
                data_collator=DataCollatorWithPadding(tokenizer=tokenizer))

trainer.train()
trainer.train(resume_from_checkpoint='test_trainer/checkpoint-30')
trainer.evaluate()

trainer.save_model('hface_model')
model.load_from_dict(torch.load('hface_model/pytorch_model.bin'))


model.eval()
for i, data in enumerate(trainer.get_eval_dataloader()):
    break

out = model(**data)
out = out['logits'].argmax(dim=1)
for i in range(8):
    print(tokenizer.decode(data['input_ids'][i], skip_special_tokens =True))
    print('label=', data['labels'][i].item())
    print('predict=', out[i].item())