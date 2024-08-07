from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path="bert-base-chinese",
                                          cache_dir=None, force_download=False)

sents = ['你站在桥上看风景',
         '看风景的人在楼上看你',
         '明月装饰了你的窗子',
         '你装饰了别人的梦']


# 一次编码单个句子或者句子对
out = tokenizer.encode(text=sents[0],text_pair=sents[1],truncation=True,
                       padding='max_length', max_length=25, add_special_tokens=True,
                       return_tensors='pt')
print(out)
print(tokenizer.decode(out))

'''
    [101, 872, 4991, 1762, 3441, 677, 4692, 7599, 3250, 102, 4692, 7599, 3250,
4638, 782, 1762, 3517, 677, 4692, 872, 102, 0, 0, 0, 0]
[CLS] 你　站 在　桥 上　看 风　景 [SEP] 看　风 景　的 人　在 楼　上 看　你 [SEP] [PAD] [PAD]
[PAD] [PAD]
'''

out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    return_tensors='pt',
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
)
for k, v in out.items():
    print(k,':', v)

tokenizer.decode(out['input_ids'])

'''
input_ids : [101, 872, 4991, 1762, 3441, 677, 4692, 7599, 3250, 102, 4692,
7599, 3250, 4638, 782, 1762, 3517, 677, 4692, 872, 102, 0, 0, 0, 0]
token_type_ids : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 0, 0, 0]
special_tokens_mask : [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 1, 1, 1]
attention_mask : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 0, 0, 0]
length : 25


'[CLS] 你　站 在　桥 上　看 风　景 [SEP] 看　风 景　的 人　在 楼　上 看　你 [SEP] [PAD] [PAD]
[PAD] [PAD]'
'''

# 批量编码成对的句子
out = tokenizer.batch_encode_plus(
    batch_text_or_text_pairs=[(sents[0], sents[1]), (sents[2], sents[3])], # batch_text_or_text_pairs=[sents[0], sents[1]]
    truncation=True,
    padding='max_length',
    max_length=25,
    add_special_tokens=True,
    return_tensors='pt',
    return_token_type_ids=True,
    return_attention_mask=True,
    return_special_tokens_mask=True,
    return_length=True
)

for k, v in out.items():
    print(k,':', v)

tokenizer.decode(out['input_ids'][0])

'''
input_ids : [[101, 872, 4991, 1762, 3441, 677, 4692, 7599, 3250, 102, 4692,
7599, 3250, 4638, 782, 1762, 3517, 677, 4692, 872, 102, 0, 0, 0, 0], 
             [101, 21128, 21129, 749, 872, 4638, 21130, 102, 872, 21129, 749, 1166, 782, 4638, 3457, 102,
0, 0, 0, 0, 0, 0, 0, 0, 0]]
token_type_ids : [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 0]]
special_tokens_mask : [[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
1, 1, 1, 1, 1, 1, 1]]
length : [21, 16]
attention_mask : [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
0, 0, 0, 0, 0]]
'[CLS] 你　站 在　桥 上　看 风　景 [SEP] 看　风 景　的 人　在 楼　上 看　你 [SEP] [PAD] [PAD]
[PAD] [PAD]'
'''

# outputs = tokenizer(examples)

vocab = tokenizer.get_vocab()
tokenizer.add_tokens(new_tokens=['明月'])
tokenizer.add_special_tokens({'eos_token':'[EOS]'})
