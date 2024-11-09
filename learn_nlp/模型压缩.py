# https://mp.weixin.qq.com/s/YwXSaw5w3XTecQXqsVRTbQ

# 量化：通过改变模型权重/激活的精度来减少内存占用
# !pip install bitsandbytes
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2' # specify the model you want to use
tokenizer = AutoTokenizer.from_pretrained(model_name)
# load the model with 8-bit quantization using bitsandbytes
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto')
input_text = "Weight Quantization is an efficient technique for compressing language models."
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
# generate text
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


# 剪枝：移除不太重要的权重或神经元，减少参数量
import torch
import torch.nn.utils.prune as prune
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'gpt2'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
'''
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2SdpaAttention(
          (c_attn): Conv1D()
          (c_proj): Conv1D()
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D()
          (c_proj): Conv1D()
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
'''

def prune_model_layer(layer, amount=0.3):
    for name, module in layer.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)

for layer in model.transformer.h:
    prune_model_layer(layer, amount=0.3)

total_params = 0
pruned_params = 0
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        total_params += module.weight.nelement()
        pruned_params += torch.sum(module.weight == 0).item()

print(f"Total parameters: {total_params}")
print(f"Pruned parameters: {pruned_params}")
print(f"Sparsity: {pruned_params / total_params:.2%}")
# Test the pruned model on a sample input
input_text = "Pruning is an effective way to compress language models."
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
# Generate text using the pruned model
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50)
# Decode and print the generated text
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


# 知识蒸馏：训练一个小模型来模仿一个大模型
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

teacher_model_name = 'gpt2'
student_model_name = 'tiny-gpt2'
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
learning_rate = 5e-5
epochs = 3
optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate)
temperature = 2.0
alpha = 0.5

for epoch in range(epochs):
    for i, example in enumerate(dataset):
        input_text = example['text']

        if not input_text.strip():
            continue

        teacher_inputs = teacher_tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
        student_inputs = student_tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=32)

        # get teacher predictions (soft labels)
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits / temperature
            teacher_probs = F.softmax(teacher_logits, dim=-1)

        student_outputs = student_model(**student_inputs)
        student_logits = student_outputs.logits

        # calculate distillation loss
        distillation_loss = F.kl_div(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (temperature ** 2)

        # calculate student task loss (cross-entropy with true labels)
        target_labels = student_inputs['input_ids']
        task_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), target_labels.view(-1), ignore_index=student_tokenizer.pad_token_id)

        loss = alpha * distillation_loss + (1-alpha) * task_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i}], Loss:{loss.item():.4f}')

# 权重共享：在不同层之间使用共享权重
import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, GPT2LMHeadModel

model = AutoModelForCausalLM.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

def apply_weight_sharing(model, num_clusters=16):
    # iterate through each parameter in the model
    for name, param in model.named_parameters():
        if param.requires_grad:
            weights = param.data.cpu().numpy().flatten().reshape(-1,1)
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(weights)
            cluster_centroids = kmeans.cluster_centers_
            labels = kmeans.labels_
            shared_weights = np.array([cluster_centroids[label] for label in labels]).reshape(param.data.shape)
            param.data = torch.tensor(shared_weights, dtype=param.data.dtype).to(param.device)
    return model



model = apply_weight_sharing(model, num_clusters=16)

