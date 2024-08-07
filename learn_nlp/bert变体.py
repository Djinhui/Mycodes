import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True) # 返回所有层隐状态
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = 'I love Paris'
tokens = tokenizer.tokenize(sentence)
tokens = ['[CLS]'] + tokens + ['[SEP]']
tokens = tokens + ['[PAD]'] + ['[PAD]']
attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
token_ids = tokenizer.convert_tokens_to_ids(tokens)
token_ids = torch.tensor(token_ids).unsqueeze(0)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)

last_hidden_state, pooler_output, hidden_states = model(token_ids, attention_mask = attention_mask)
'''
last_hidden_state包含从最后的编码器中获得的所有标记的特征。
pooler_output表示来自最后的编码器的[CLS]标记的特征,它被一个线性激活函数和tanh激活函数进一步处理。
hidden_states包含从所有编码器层获得的所有标记的特征
'''
print(last_hidden_state.shape) # (1, 7, 768)
print(pooler_output.shape) # (1, 768)
print(len(hidden_states)) # 13 = 嵌入层+12个编码器层输出
print(hidden_states[0].shape) # (1, 7, 768)
print(hidden_states[12].shape) # (1, 7, 768)


# 1. ---------------ALBERT-------------
from transformers import AlbertModel, AlbertTokenizer

model = AlbertModel.from_pretrained('albert-base-v2')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

sentence = 'Paris is a beautiful city'
inputs = tokenizer(sentence, return_tensors='pt')
print(inputs)
'''
{
'input_ids': tensor([[   2, 1162,   25,   21, 1632,  136,    3]]),
'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0]]),
'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])
}
'''

last_hidden_state, cls_head = model(**inputs)
print(last_hidden_state.shape) # (1, 7, 768)
print(cls_head.shape) # (1, 768)

# 2. ---------------RoBERTa------------------
from transformers import RobertaModel, RobertaTokenizer
model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
print(model.config)


# 3. ---------------ELECTRA--------------
from transformers import ElectraModel, ElectraTokenizer
model = ElectraModel.from_pretrained('google/electra-base-discriminator') # 判别器
model = ElectraModel.from_pretrained('google/electra-base-generator') # 生成器

tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
print(model.config)

# 4. ----------------SpanBERT-------------
from transformers import pipeline
qa_pipeline = pipeline("question-answering",
                       model="mrm8488/spanbert-large-finetuned-squadv2",
                       tokenizer="SpanBERT/spanbert-large-cased")

results = qa_pipeline({'question': "What is machine learning?",
                       'context': "Machine learning is a subset of artificial intelligence. \
                        It is widely for creating a variety of applications such as email filtering and computer vision"
})

print(results['answer']) # a subset of artificial intelligence