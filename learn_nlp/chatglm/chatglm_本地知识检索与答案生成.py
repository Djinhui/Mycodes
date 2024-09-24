# 《从零开始大模型开发与微调》CHP17-基于知识链的ChatGLM本地化知识库检索与智能答案生成
'''
知识链问答流程:
本地知识文件(txt-json-pdf-doc)->读取内容->文本切分->构成Documents->Embedding->得到每个document的Embedding->构建索引
输入Query->Embedding->从本地知识构建的索引进行搜索->得到Top k相关的Document->将Document拼接得到Context->用Context和Query填充Prompt模板，得到Prompt->输入到LLM->得到Response

这里实际上是将用户请求的Query和Document进行匹配，也就是所谓的问题－文档匹配.
问题－文档匹配的问题在于问题和文档在表达方式上存在较大差异。通常Query以疑问句为主，而Document则以陈述说明为主，这种差异可能会影响最终匹配的效果

一种改进方法是，跳过问题和文档匹配部分，先通过Document生成一批候选的问题－答案匹配，当用户发来请求的时候，
首先是把Query和候选的Question进行匹配，进而找到相关的Document片段，此时的具体思路如下：

首先准备好文档，并整理为纯文本的格式，把每个文档切成若干个小的模块。
调用ChatGLM的API，根据每个模块生成5个候选的Question，使用的Prompt格式为“请根据下面的文本生成5个问题：……”​。
调用文本转向量的接口，将生成的Question转为向量，存入向量数据库，记录Question和原始模块的对应关系。
当用户发来一个问题的时候，将问题同样转为向量，并检索向量数据库，得到相关性最高的一个Question，进而找到对应的模块。
将问题和模块合并重写为一个新的请求发给ChatGLM进行文档问答

'''

import torch
from transformers import BertModel, BertTokenizer
import os
import numpy as np
from typing import List
from rank_bm25 import BM25Okapi


embedding_model_name = 'shibing624/text2vec-base-chinese'
embedding_model_length = 512
tokenizer = BertTokenizer.from_pretrained(embedding_model_name)
model = BertModel.from_pretrained(embedding_model_name)


def get_top_n_sim_text(query:str, documents:List[str], top_n=1):
    tokenized_corpus = []
    for doc in documents:
        text = []
        for char in doc:
            text.append(char)
        tokenized_corpus.append(text)

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = [char for char in query]
    results = bm25.get_top_n(tokenized_query, tokenized_corpus,n=top_n)
    results = ["".join(res) for res in results]
    return results


def generate_prompt(question:str, relevant_chunks:List[str]):
    prompt = f'根据文档内容来回答问题，问题是"{question}"， 文档内容如下:\n'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt


def strict_generate_prompt( question: str, relevant_chunks: List[str]):
    prompt = f'严格根据文档内容来回答问题，回答不允许编造成分要符合原文内容，问题是"{question}"，文档内容如下：\n'
    for chunk in relevant_chunks:
        prompt += chunk + "\n"
    return prompt


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expand = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings*input_mask_expand, 1) / torch.clamp(input_mask_expand.sum(1), min=1e-9)

def compute_sim_score( v1: np.ndarray, v2: np.ndarray) -> float:
        return v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# -----------------------------本地知识索引构建----------------------------
sentences = []
path = './dataset/financial_research_reports' # 本地知识文档
filelist = [path + i for i in os.listdir(path)]
for file in filelist:
    if file.endswith('txt'):
        with open(file, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            lines = "".join(lines)
            sentences.append(lines[:embedding_model_length])

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
sentence_embeddings_np = sentence_embeddings.detach().numpy()
np.save("sentence_embeddings_np.npy", sentence_embeddings_np)


# --------------------------相关索引搜索-----------------
query = ['雅生活服务的人工成本占营业成本的比例是多少']
query_input = tokenizer(query, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    model_output = model(**query_input)
query_embedding = mean_pooling(model_output, query_input['attention_mask'])
print(query_embedding.shape) #(1, 768)

sentence_embeddings_np = np.load('sentence_embeddings_np.npy')

for i in range(len(sentence_embeddings_np)):
     score = compute_sim_score(sentence_embeddings_np[i], query_embedding[0])
     print(f'The score for document {i} similar with query is {score:.2f}')

# ---------------------------构建Prompt----------------------
file = "./dataset/financial_research_reports/yanbao001.txt" # 与query最相关的本地文档
context_list = []
with open(file,"r",encoding="UTF-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        context_list.append(line)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b",trust_remote_code=True).half().cuda()

print('普通ChatGLM询问结果:')
query = ['雅生活服务的人工成本占营业成本的比例是多少']
prompt = query[0]
respose, history = model.chat(tokenizer, prompt, history=[])
print(respose)


print('查询与问题相似的文档后一起输入模型进行询问结果:')
sim_results = get_top_n_sim_text(query=query[0], documents=context_list)
prompt = strict_generate_prompt(query[0], sim_results)
response, _ = model.chat(tokenizer, prompt, history=[])
print(response)