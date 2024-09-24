import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training,get_peft_model_state_dict
import modeling_chatglm
import prepare_data_chatglm
from accelerate import Accelerator


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = modeling_chatglm.JinhuiModel(model_path='../huggingface_saver/chatglm6b.pth')

peft_config = LoraConfig.from_pretrained('./peft')
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model = model.half().to(device)

prompt_text = "按给定的格式抽取文本信息。\n文本:"
tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
all_train_data = prepare_data_chatglm.get_train_data('./data/spo_0.json', tokenizer, 32, 48, prompt_text)
train_dataset = prepare_data_chatglm.Seq2SeqDataset(all_train_data)
train_loader = DataLoader(train_dataset, batch_size=2, drop_last=True, collate_fn=prepare_data_chatglm.coll_fn, num_workers=0)

accelerator = Accelerator()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9,0.999), eps=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2400, eta_min=2e-6, last_epoch=-1)

model, optim, train_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

for epoch in range(20):
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
    if (epoch +1) %3 == 0:
        torch.save(model.state_dict(), "./glm6b_peft.pth")
