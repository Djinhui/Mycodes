import torch
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, DistilBertForSequenceClassification
import textbrewer
from textbrewer import GeneralDistiller, TrainingConfig, DistillationConfig

dataset = load_dataset('glue', 'sst2', split='train')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['sentence'], padding='max_length', truncation=True)

dataset = dataset.map(tokenize_function, batched=True)
encoded_dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

def collate_fn(examples):
    return dict(tokenizer.pad(examples, return_tensors='pt'))

dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=8, collate_fn=collate_fn)

teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
student_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

print("teacher_model's params")
result,_ = textbrewer.utils.display_parameters(teacher_model, max_level=3)
print(result)

print("student_model's params")
result,_ = textbrewer.utils.display_parameters(student_model, max_level=3)
print(result)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    teacher_model = teacher_model.cuda()
    student_model = student_model.cuda()

def simple_adaptor(batch, model_outputs):
    return {'logits':model_outputs[1]}

train_config = TrainingConfig(device=device,)
distill_config = DistillationConfig()

# 定义distiller
distiller = GeneralDistiller(
    train_config=train_config, distill_config=distill_config,
    model_T=teacher_model, model_S=student_model,
    adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)

# 开始蒸馏！
with distiller:
    distiller.train(
        optimizer, dataloader,
        scheduler_class=None, scheduler_args=None,
        num_epochs=1, callback=None)
