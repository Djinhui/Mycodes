from sentence_transformers import SentenceTransformer, util
import numpy as np


# 使用Sentence-BERT计算句子特征
model = SentenceTransformer('bert-base-nli-ean-tokens')
sentence = 'beijing is a befautiful city'
sentence_representation = model.encode(sentence)
print(sentence_representation.shape) # (768,)

# 计算两个句子的相似度
sentence1 = 'it is a great day'
sentence2 = 'today is awesome'
sentence1_representation = model.encode(sentence1)
sentence2_representation = model.encode(sentence2)
similarity = util.pytorch_cos_sim(sentence1_representation, sentence2_representation)
print(similarity) # tensor([[0.9313]])

# 加载自定义模型
from sentence_transformers import SentenceTransformer, models

word_embedding_model = models.Transformer('albert-base-v2') # 返回每个token特征
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)  # 定义特征汇聚模式

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
sentence = 'beijing is a befautiful city'
sentence_representation = model.encode(sentence)
print(sentence_representation.shape) # (768,)

# 用Sentence-BERT寻找相似句子
model = SentenceTransformer('bert-base-nli-mean-tokens')

master_dict = [
                 'How to cancel my order?',
                 'Please let me know about the cancellation policy?',
                 'Do you provide a refund?',
                 'what is the estimated delivery date of the product?',
                 'why my order is missing?',
                 'how do i report the delivery of the incorrect items?'
                 ]

input_question = 'When is my product getting delivered?'
input_question_representation = model.encode(input_question, convert_to_tensor=True)
master_dict_representation = model.encode(master_dict, convert_to_tensor=True)
similarity = util.pytorch_cos_sim(input_question_representation, master_dict_representation)

print('The most similar question in the master dictionary to given input question is:\n',
      master_dict[np.argmax(similarity)])



