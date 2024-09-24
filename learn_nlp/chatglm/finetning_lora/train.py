import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from huggingface_saver import configuration_chatglm, modeling_chatglm
import jinhui_model
from tqdm import tqdm
import prepare_data_chatglm
from accelerate import Accelerator
from minlora.model import *
from minlora.utils import *
import prepare_data_chatglm
from accelerate import Accelerator


config = configuration_chatglm.ChatGLMConfig()
model = jinhui_model.JinhuiModel(model_path='./huggingface_saver/chatgbm6b.pth', config=config, strict=False)
model = model.half().cuda()

for name, param in model.named_parameters():
    param.requires_grad = False

for key, _layer in model.named_modules():
    if 'query_key_value' in key:
        add_lora(_layer)

jinhui_model.print_trainable_parameters(model)


prompt_text = "按给定的格式抽取文本信息。\n文本:"#"你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
all_train_data = prepare_data_chatglm.get_train_data('./data/spo_0.json', tokenizer, 32, 48, prompt_text)
train_dataset = prepare_data_chatglm.Seq2SeqDataset(all_train_data)
train_loader = DataLoader(train_dataset, batch_size=2, drop_last=True, collate_fn=prepare_data_chatglm.coll_fn, num_workers=0)

accelerator = Accelerator()

lora_parameters = [{"params":list(get_lora_params(model))}]
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# optimizer = torch.optim.AdamW(lora_parameters.parameters(), lr=2e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2400, eta_min=2e-6, last_epoch=-1)
model, optim, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)
for epoch in range(96):
    pbar = tqdm(train_loader, total=len(train_loader))
    for batch in (pbar):
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()

        _,_,loss = model.forward(input_ids,labels=labels)
        accelerator.backward(loss)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        lr_scheduler.step()  # 执行优化器
        optimizer.zero_grad()

        pbar.set_description(
            f"epoch:{epoch + 1}, train_loss:{loss.item():.5f}, lr:{lr_scheduler.get_last_lr()[0] * 1000:.5f}")
        
torch.save(model.state_dict(), './glm6b_lora_all.pth')
lora_state_dict = get_lora_state_dict(model)
torch.save(lora_state_dict, '/glm6b_lora_only.pth')



# ----------------------------------pred----------------------------
import time
import json

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

config = configuration_chatglm.ChatGLMConfig()
model = jinhui_model.JinhuiModel(model_path='./huggingface_saver/chatglm6b.pth', config=config, strict=False)

for key, layer in model.named_modules():
    if 'query_key_value' in key:
        add_lora(layer)
for name, param in model.named_parameters():
    param.requires_grad = False

model.load_state_dict(torch.load('./glm6b_lora_only.pth'), strict=False)
model = model.half().cuda()

jinhui_model.print_trainable_parameters(model)
model.eval()
max_len = 288
max_src_len = 256
#prompt_text = "你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本："
prompt_text = "按给定的格式抽取文本信息。\n文本:"
save_data = []
f1 = 0.0
max_tgt_len = max_len - max_src_len - 3
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)

s_time = time.time()
with open('./data/spo_0.json', 'r', encoding='utf-8') as fh:
    for i, line in enumerate(tqdm(fh, desc='iter')):
        with torch.no_grad():
            sample = json.load(line.strip())
            src_tokens = tokenizer.tokenize(sample['text'])
            prompt_tokens = tokenizer.tokenize(prompt_text)

            if len(src_tokens) > max_src_len - len(prompt_tokens):
                src_tokens = src_tokens[:max_src_len - len(prompt_tokens)]

            tokens = prompt_tokens + src_tokens + ["[gMAKS]", "<sop>"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # input_ids = tokenizer.encode('帮我xxxxx')

            for _ in range(max_tgt_len):
                input_ids_tensor = torch.tensor([input_ids]).to("cuda")
                logits, _, _ = model.forward(input_ids_tensor)
                logits = logits[:, -3]
                probs = torch.softmax(logits / 0.95, dim=-1)
                next_token = sample_top_p(probs, 0.95)  # 预设的top_p = 0.95
                # next_token = next_token.reshape(-1)

                # next_token = result_token[-3:-2]
                input_ids = input_ids[:-2] + [next_token.item()] + input_ids[-2:]
                if next_token.item() == 130005:
                    print("break")
                    break
            result = tokenizer.decode(input_ids)
            print(result)
            print("---------------------------------")