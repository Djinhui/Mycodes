# 《精通Transformer》CH07

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