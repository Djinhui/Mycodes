# coding:utf-8
import json
import torch
from torch.utils.data import DataLoader,Dataset
from random import choice
from config import *
from collections import defaultdict
conf = Config()



def find_head_idx(source,target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1

def collate_fn(data):
    text_list = [value[0] for value in data]
    triple = [value[1] for value in data]
    text = conf.tokenizer.batch_encode_plus(text_list,padding=True,max_length=conf.max_length, truncation=True)
    batch_size = len(text['input_ids'])
    seq_len = len(text['input_ids'][0])

    sub_heads = []
    sub_tails = []
    obj_heads = []
    obj_tails = []
    sub_len = []
    sub_head2tail = []
    for batch_index in range(batch_size):
        inner_input_ids = text['input_ids'][batch_index]
        inner_triples = triple[batch_index]
        results = create_label(inner_triples,inner_input_ids,seq_len)
        sub_len.append(results[0])
        sub_head2tail.append(results[1])
        sub_heads.append(results[2])
        sub_tails.append(results[3])
        obj_heads.append(results[4])
        obj_tails.append(results[5])
    input_ids = torch.tensor(text['input_ids']).to(conf.device)
    mask = torch.tensor(text['attention_mask']).to(conf.device)
    sub_heads = torch.stack(sub_heads).to(conf.device)
    sub_tails = torch.stack(sub_tails).to(conf.device)
    sub_len = torch.stack(sub_len).to(conf.device)
    sub_head2tail = torch.stack(sub_head2tail).to(conf.device)
    obj_heads = torch.stack(obj_heads).to(conf.device)
    obj_tails = torch.stack(obj_tails).to(conf.device)

    inputs = {'input_ids':input_ids,
              'mask':mask,
              'sub_head2tail':sub_head2tail,
              'sub_len':sub_len}
    labels = {'sub_heads':sub_heads,
              'sub_tails':sub_tails,
              'obj_heads':obj_heads,
              'obj_tails':obj_tails}

    return inputs,labels

def create_label(inner_triples,inner_input_ids,seq_len):
    inner_sub_heads,inner_sub_tails = torch.zeros(seq_len),torch.zeros(seq_len)
    inner_sub_head,inner_sub_tail = torch.zeros(seq_len),torch.zeros(seq_len)
    inner_obj_heads = torch.zeros((seq_len,conf.num_rel))
    inner_obj_tails = torch.zeros((seq_len,conf.num_rel))
    inner_sub_head2tail = torch.zeros(seq_len)

    inner_sub_len = torch.tensor([1], dtype=torch.float)
    s2ro_map = defaultdict(list)

    for inner_triple in inner_triples:
        # 将每个三元组中的主语、谓语和宾语转化为对应的token_id序列
        inner_triple = (conf.tokenizer(inner_triple['subject'],add_special_tokens=False)['input_ids'],
                        conf.rel_vocab.to_index(inner_triple['predicate']),
                        conf.tokenizer(inner_triple['object'],add_special_tokens=False)['input_ids']
                        )

        sub_head_idx = find_head_idx(inner_input_ids,inner_triple[0])
        obj_head_idx = find_head_idx(inner_input_ids,inner_triple[2])

        # 如果主语和宾语在输入文本序列中都能找到，则更新标签信息
        if sub_head_idx != -1 and obj_head_idx != -1:
            sub = (sub_head_idx,sub_head_idx+len(inner_triple[0]) - 1)
            s2ro_map[sub].append(
                (obj_head_idx,obj_head_idx+len(inner_triple[2]) - 1,inner_triple[1])
            )
    if s2ro_map:
        for s in s2ro_map:
            inner_sub_heads[s[0]] = 1
            inner_sub_tails[s[1]] = 1

        sub_head_idx,sub_tail_idx = choice(list(s2ro_map.keys()))
        inner_sub_head[sub_head_idx] = 1
        inner_sub_tail[sub_tail_idx] = 1
        inner_sub_head2tail[sub_head_idx:sub_tail_idx + 1] = 1
        inner_sub_len = torch.tensor([sub_tail_idx + 1 - sub_head_idx],dtype=torch.float)

        for ro in s2ro_map.get((sub_head_idx,sub_tail_idx),[]):
            inner_obj_heads[ro[0]][ro[2]] = 1
            inner_obj_tails[ro[1]][ro[2]] = 1

    return inner_sub_len,inner_sub_head2tail,inner_sub_heads,inner_sub_tails,inner_obj_heads,inner_obj_tails

class MyDataset(Dataset):
    def __init__(self,data_path):
        super(MyDataset, self).__init__()
        self.dataset = [json.loads(line) for line in open(data_path,encoding='utf-8')]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        content = self.dataset[index]
        text = content['text']
        spo_list = content['spo_list']
        return text,spo_list

def get_loader():
    train_data = MyDataset(conf.train_data_path)
    dev_data = MyDataset(conf.dev_data_path)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(dataset=dev_data,
                                batch_size=conf.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn)
    print(dev_dataloader)
    return train_dataloader,dev_dataloader



if __name__ == '__main__':
    get_loader()