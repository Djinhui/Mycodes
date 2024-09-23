import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# from transformers.models.marian import MarianPreTrainedModel, MarianModel
from transformers import AdamW, get_scheduler
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm
import json

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed_everything(seed=42)


max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

max_input_length = 128
max_target_length = 128

batch_size = 32
learning_rate = 1e-5
epoch_num = 3


class TRANS(Dataset):
    """
    选择 translation2019zh 语料作为数据集，它共包含中英文平行语料 520 万对
    {"english": "In Italy, there is no real public pressure for a new, fairer tax system.", "chinese": "在意大利，公众不会真的向政府施压，要求实行新的、更公平的税收制度。"}
    """
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample

        return Data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

data = TRANS('translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('translation2019zh/translation2019zh_valid.json')

model_checkpoint= "Helsinki-NLP/opus-mt-zh-en" # Marian Model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = model.to(device)

def collate_fn(batch_examples):
    batch_inputs, batch_targets = [], []
    for sample in batch_examples:
        batch_inputs.append(sample['chinese'])
        batch_targets.append(sample['english'])
    
    batch_data = tokenizer(
        batch_inputs, padding=True,
        max_length=max_input_length, truncation=True, return_tensors='pt'
    )
    '''
    batch_data = tokenizer(
        batch_inputs, 
        text_target=batch_targets, 
        padding=True, 
        max_length=max_length,
        truncation=True, 
        return_tensors="pt"
    )
    '''
    # 默认情况下分词器会采用源语言的设定来编码文本，要编码目标语言则需要通过上下文管理器 as_target_tokenizer()
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch_targets,
                           padding=True,
                           max_length=max_target_length,
                           truncation=True,
                           return_tensors='pt')['input_ids']
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels) # decoder_input:[<s> i love you], decoder_label:[i love you </s>]
        end_token_index = torch.where(labels==tokenizer.eos_token_id)[1]
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1] = -100 # ignore pad loss
        batch_data['labels'] = labels

    return batch_data # dict_keys(['input_ids', 'attention_mask', 'decoder_input_ids', 'labels'])


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss:{0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss:{total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss

bleu = BLEU()

def test_loop(dataloader, model):
    preds, labels = [], []
    model.eval()
    for batch_data in tqdm(dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=max_target_length
            ).cpu().numpy()
        label_tokens = batch_data['labels'].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]

    blue_score = bleu.corpus_score(preds, labels).score
    print(f'BLUE:{blue_score:>0.2f}')
    return blue_score


optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_training_steps=epoch_num*len(train_dataloader),
    num_warmup_steps=0
)

# 在开始训练之前，我们先评估一下没有微调的预训练模型在测试集上的性能
test_loop(test_dataloader, model) # 42.61

total_loss = 0.0
best_blue = 0.0
for t in range(epoch_num):
    print(f"Epoch {t+1} / {epoch_num}")
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, t+1, total_loss)
    valid_blue = test_loop(valid_dataloader, model)
    if valid_blue > best_blue:
        best_blue = valid_blue
        print('saving new weights.........')
        torch.save(model.state_dict(),
                   f'epoch_{t+1}_valid_blue_{valid_blue:0.2f}_model_weights.bin')
        

print('Done!')


# 测试模型
model.load_state_dict(torch.load('epoch_1_valid_bleu_53.38_model_weights.bin'))
model.eval()
with torch.no_grad():
    sources, preds, labels = [], [], []
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(
            batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            max_length=max_target_length
        ).cpu().numpy()
        label_tokens = batch_data['labels'].cpu().numpy()

        decoded_sources = tokenizer.batch_decode(
            batch_data['input_ids'].cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True
        )
        
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        sources += [source.strip() for source in sources]
        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]

    bleu_score = bleu.corpus_score(preds, labels).score()
    print(f'Test set BLUE:{bleu_score:>0.2f}')
    results = []
    print('saving translate results')
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            'sentence':source,
            'prediction':pred,
            'tranlation':label[0]
        })
    
    with open('test_translated.json', 'wt', encoding='utf-8') as f:
        for example in results:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')


# 关于解码
# Transformers 库中所有的生成模型都提供了用于自回归生成的 generate() 函数，例如 GPT2、XLNet、OpenAi-GPT、CTRL、TransfoXL、XLM、Bart、T5 等等
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# # add the EOS token as PAD token to avoid warnings
# model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

# # 贪心搜索 Greedy Search
# input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
# greedy_output = model.generate(input_ids, max_length=50)
# print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# 集束搜索 Beam Search
# beam_output = model.generate(
#     input_ids, 
#     max_length=50, 
#     num_beams=5, 
#     early_stopping=True
# )

# beam_output = model.generate(
#     input_ids, 
#     max_length=50, 
#     num_beams=5, 
#     no_repeat_ngram_size=2, 
#     early_stopping=True
# )
# print(tokenizer.decode(beam_output[0], skip_special_tokens=True))

# beam_outputs = model.generate(
#     input_ids, 
#     max_length=50, 
#     num_beams=5, 
#     no_repeat_ngram_size=2, 
#     num_return_sequences=3, 
#     early_stopping=True
# )

# for i, beam_output in enumerate(beam_outputs):
#     print("{}: {}\n\n".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))


# 随机采样
# sample_output = model.generate(
#     input_ids, 
#     do_sample=True, 
#     max_length=50, 
#     top_k=0
# )

# sample_output = model.generate(
#     input_ids, 
#     do_sample=True, 
#     max_length=50, 
#     top_k=0, 
#     temperature=0.6
# )

# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# # Top-K采样
# sample_output = model.generate(
#     input_ids, 
#     do_sample=True, 
#     max_length=50, 
#     top_k=10
# )
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# # Top-p采样
# sample_output = model.generate(
#     input_ids, 
#     do_sample=True, 
#     max_length=50, 
#     top_p=0.92, 
#     top_k=0
# )
# print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# # Top-K Top-p
# sample_outputs = model.generate(
#     input_ids,
#     do_sample=True, 
#     max_length=50, 
#     top_k=50, 
#     top_p=0.95, 
#     num_return_sequences=3
# )
# for i, sample_output in enumerate(sample_outputs):
#     print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))