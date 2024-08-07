# 《精通Transformer》CH07

# 1. 句子相似性
from datasets import load_dataset, load_metric
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
import math
import tensorflow as tf

metric = load_metric('glue', 'mrpc')
mrpc = load_dataset('glue','mrpc')

metric = load_metric('glue', 'stsb') # 语义文本相似基准
stsb_metric = load_metric('glue', 'stsb')  # 斯皮尔曼相关系数值和皮尔逊相关系数值
stsb = load_dataset('glue', 'stsb') 

use_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
distilroberta = SentenceTransformer('stsb-distilroberta-base-v2')

def use_sts_benchmark(batch):
    sts_encode1 = tf.nn.l2_normalize(use_model(batch['sentence1']), axis=1)
    sts_encode2 = tf.nn.l2_normalize(use_model(batch['sentence2']), axis=1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
    scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi
    return scores

def roberta_sts_benchmark(batch): 
  sts_encode1 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence1']),axis=1) 
  sts_encode2 = tf.nn.l2_normalize(distilroberta.encode(batch['sentence2']),axis=1) 
  cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1,sts_encode2),axis=1) 
  clip_cosine_similarities = tf.clip_by_value(cosine_similarities,-1.0,1.0) 
  scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi 
  return scores 

use_results = use_sts_benchmark(stsb['validation']) 
distilroberta_results = roberta_sts_benchmark(stsb['validation']) 

references = [item['label'] for item in stsb['validation']]
results = { 
      "USE":stsb_metric.compute( 
                predictions=use_results, 
                references=references), 
      "DistillRoberta":stsb_metric.compute( 
                predictions=distilroberta_results, 
                references=references) 
} 

import pandas as pd 
pd.DataFrame(results) 

# 2. 使用BART进行零样本学习文本分类
# 使用标签和上下文之间的语义相似性来执行零样本分类
# eg. x:one day i will see the world  候选labels:['travel', 'exploration', 'dancing','cooking']
# one day i will see the world 与 'travel'的相似性最高，分到'travel'类

import pandas as pd
from transformers import pipeline

# 2.1 BART pipeline
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
sequence_to_classify = 'one day i will see the world'
candidate_labels = ['travel', 'exploration', 'dancing','cooking']
result = classifier(sequence_to_classify, candidate_labels)
print(pd.DataFrame(result))

'''
sequence	                        labels	scores
0	one day I will see the world	travel	0.795756
1	one day I will see the world	exploration	0.199332
2	one day I will see the world	dancing	0.002621
3	one day I will see the world	cooking	0.002291
'''

result = classifier(sequence_to_classify,  
                      candidate_labels,  
                      multi_label=True) # 可以属于多个类
pd.DataFrame(result) 

# 2.2 BART no pipeline
'''
BART在自然语言推理(Natural Language Inference，NLI)数据集（如多体裁自然语言推理）上进行了微调。
这些数据集包含句子对，每对句子属于如下3个类别：Neutral（中立）、Entailment（蕴含）和Contradiction（矛盾

'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

premise = "one day I will see the world" 
label = "travel" 
hypothesis = f'This example is {label}.' 

x = x = tokenizer.encode( 
    premise, 
    hypothesis, 
    return_tensors='pt', 
    truncation_strategy='only_first') 

logits = model(x)[0] 
entail_contradiction_logits = logits[:,[0,2]] 
probs = entail_contradiction_logits.softmax(dim=1) 
prob_label_is_true = probs[:,1] 
print(prob_label_is_true) # tensor([0.9945], grad_fn=<SelectBackward>)

# 3. 使用FLAIR库进行语义相似性
import pandas as pd
import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from flair.embeddings import TransformerWordEmbeddings,TransformerDocumentEmbeddings
from flair.embeddings import DocumentRNNEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings

similar=[("A black dog walking beside a pool.","A black dog is walking along the side of a pool."),
("A blonde woman looks for medical supplies for work in a suitcase.	"," The blond woman is searching for medical supplies in a suitcase."),
("A doubly decker red bus driving down the road.","A red double decker bus driving down a street."),
("There is a black dog jumping into a swimming pool.","A black dog is leaping into a swimming pool."),
("The man used a sword to slice a plastic bottle.	","A man sliced a plastic bottle with a sword.")]
pd.DataFrame(similar, columns=["sen1", "sen2"])

dissimilar= [("A little girl and boy are reading books. ", "An older child is playing with a doll while gazing out the window."),
("Two horses standing in a field with trees in the background.", "A black and white bird on a body of water with grass in the background."),
("Two people are walking by the ocean." , "Two men in fleeces and hats looking at the camera."),
("A cat is pouncing on a trampoline.","A man is slicing a tomato."),
("A woman is riding on a horse.","A man is turning over tables in anger.")]
pd.DataFrame(dissimilar, columns=["sen1", "sen2"])

def sim(s1, s2):
   s1 = s1.embedding.unsqueeze(0)
   s2 = s2.embedding.unsqueeze(0)
   sim = torch.cosine_similarity(s1, s2).item()
   return np.round(sim)

def evaluate(embeddings, myPairList):
   scores = []
   for s1, s2 in myPairList:
      s1, s2 = Sentence(s1), Sentence(s2)
      embeddings.embed(s1)
      embeddings.embed(s2)
      scores.append(sim(s1, s2))
   return scores, np.round(np.mean(scores))

# 3.1 平均词嵌入
'''
平均词嵌入［或者称为文档池(document pooling)］将平均池操作应用于句子中的所有单词，其中所有单词嵌入的平均值被视为句子嵌入。
'''
glove_embedding = WordEmbeddings('glove')
glove_pool_embeddings = DocumentPoolEmbeddings([glove_embedding])
glove_scores, glove_mean = evaluate(glove_pool_embeddings, similar)
evaluate(glove_pool_embeddings, dissimilar)

# 3.2 基于RNN的文档嵌入
gru_embeddings = DocumentRNNEmbeddings([glove_embedding])
evaluate(gru_embeddings, similar)
evaluate(gru_embeddings, dissimilar)

# 3.3 基于Transformer的BERT嵌入
bert_embeddings = TransformerDocumentEmbeddings('bert-base-uncased')
evaluate(bert_embeddings, similar)
evaluate(bert_embeddings, dissimilar)

# 3.4 Sentence-BERT 嵌入
sbert_embeddings = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')
evaluate(sbert_embeddings, similar)
evaluate(sbert_embeddings, dissimilar)

tricky_pairs=[("An elephant is bigger than a lion","A lion is bigger than an elephant"),
              ("the cat sat on the mat","the mat sat on the cat")]

evaluate(glove_pool_embeddings, tricky_pairs)
evaluate(gru_embeddings, tricky_pairs)
evaluate(bert_embeddings, tricky_pairs)
evaluate(sbert_embeddings, tricky_pairs)


# 4. 基于SentenceBERT的文本聚类
# 4.1 基于paraphrase-distilroberta-base-v1的主题建模
import pandas as pd, numpy as np
import torch, os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import scipy
import matplotlib.pyplot as plt
import umap


dataset = load_dataset("amazon_polarity",split="train")
corpus = dataset.shuffle(seed=42)[:10000]['content']
pd.Series([len(e.split()) for e in corpus]).hist()

model_path="paraphrase-distilroberta-base-v1"
#paraphrase-distilroberta-base-v1 - Trained on large scale paraphrase data.
model = SentenceTransformer(model_path)
corpus_embeddings = model.encode(corpus)
corpus_embeddings.shape # (10000, 768)

kmeans = KMeans(n_clusters=5, random_state=0).fit(corpus_embeddings)
cls_dist = pd.Series(kmeans.labels_).value_counts()

# 这里假设最接近质心的句子是对应于聚类中最具代表性的样例。
distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_ , 
                                         corpus_embeddings)

centers={}
print("Cluster", "Size", "Center-idx", "Center-Example", sep="\t\t")
for i,d in enumerate(distances):
  ind = np.argsort(d, axis=0)[0]
  centers[i]=ind
  print(i,cls_dist[i], ind, corpus[ind] ,sep="\t\t")

X = umap.UMAP(n_components=2, min_dist=0.0).fit_transform(corpus_embeddings)
labels= kmeans.labels_

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(X[:,0], X[:,1], c=labels, s=1, cmap='Paired')
for c in centers:
    plt.text(X[centers[c],0], X[centers[c], 1], "CLS-"+ str(c), fontsize=24) 
plt.colorbar()

# 4.2 基于BERTopic的主题建模
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer("paraphrase-distilroberta-base-v1")
topic_model = BERTopic(embedding_model=sentence_model)
topics, _ = topic_model.fit_transform(corpus)

topic_model.get_topic_info()[:6]
topic_model.get_topic(2)


# 5. 基于Sentence-BERT的语义搜索
import pandas as pd
import sklearn
import numpy as np
import scipy
from sentence_transformers import SentenceTransformer


wwf_faq=["I haven’t received my adoption pack. What should I do?",
         "How quickly will I receive my adoption pack?",
         "How can I renew my adoption?",
         "How do I change my address or other contact details?",
         "Can I adopt an animal if I don’t live in the UK?",
         "If I adopt an animal, will I be the only person who adopts that animal?",
"My pack doesn't contain a certicate",
"My adoption is a gift but won’t arrive on time. What can I do?",
"Can I pay for an adoption with a one-off payment?",
"Can I change the delivery address for my adoption pack after I’ve placed my order?",
"How long will my adoption last for?",
"How often will I receive updates about my adopted animal?",
"What animals do you have for adoption?",
"How can I nd out more information about my adopted animal?",
"How is my adoption money spent?",
"What is your refund policy?",
"An error has been made with my Direct Debit payment, can I receive a refund?",
"How do I change how you contact me?"]

test_questions=["What should be done, if the adoption pack did not reach to me?",
                " How fast is my adoption pack delivered to me?",
                "What should I do to renew my adoption?",
        "What should be done to change adress and contact details ?",
      "I live outside of the UK, Can I still adopt an animal?"]



model = SentenceTransformer("quora-distilbert-base")

faq_embeddings = model.encode(wwf_faq)
test_q_emb= model.encode(test_questions)

# 度量每个测试问题和FAQ中每个问题的相似性
from scipy.spatial.distance import cdist

for q, qe in zip(test_questions, test_q_emb):
    distances = cdist([qe], faq_embeddings, "cosine")[0]
    ind = np.argsort(distances, axis=0)[:3]
    print("\n Test Question: \n "+q)
    for i,(dis,text) in enumerate(zip(distances[ind], [wwf_faq[i] for i in ind])):
        print(dis,ind[i],text, sep="\t")

def get_best(query, K=3):
    query_embedding = model.encode([query])
    distances = cdist(query_embedding, faq_embeddings, "cosine")[0]
    ind = np.argsort(distances, axis=0)
    print("\n"+query)
    for c,i in list(zip(distances[ind],  ind))[:K]:
        print(c,wwf_faq[i], sep="\t")

get_best("How do I change my contact info?",3)

'''

How do I change my contact info?
0.056767916805916196	How do I change my address or other contact details?
0.18566553083561854	How do I change how you contact me?
0.324083301839059	How can I renew my adoption?
'''