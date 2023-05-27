import jieba
import numpy as np
from keras.preprocessing.text import Tokenizer

# 1 jieba分词实现
text_samples = ["我爱中国", "爸爸妈妈爱我", "爸爸妈妈爱中国"]
# 先中文进行分词
tokens_samples = []
for text in text_samples:
    jiebas = jieba.lcut(text)
    tokens_samples.append(jiebas)

# 构建词的索引
tokens_samples = ['爸爸 妈妈 爱 我', '爸爸 妈妈 爱 中国', '我 爱 中国']
token_index = {}
for sample in tokens_samples:
    for word in sample.split(' '):
        if word not in token_index:
            token_index[word] = len(token_index) + 1

# 构建one-hot编码矩阵
results = np.zeros(shape=(len(token_index), max(token_index.values())+1))
for i, sample in enumerate(tokens_samples):
    for _, word in list(enumerate(sample.split())):
        index = token_index[word]
        results[i, index] = 1


# 2. Keras实现
from keras.preprocessing.text import Tokenizer
tokens_samples = ['爸爸 妈妈 爱 我', '爸爸 妈妈 爱 中国', '我 爱 中国']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens_samples)
# tokenizer.fit_on_sequences()
word_index = tokenizer.word_index
print(word_index)  # {'爱': 1, '爸爸': 2, '妈妈': 3, '我': 4, '中国': 5}
print(len(word_index))  # 5

# 将词替换成索引
sequences = tokenizer.texts_to_sequences(tokens_samples)
print(sequences)  # [[2, 3, 1, 4], [2, 3, 1, 5], [4, 1, 5]]

# 构建one-hot编码
onehot_mat = tokenizer.texts_to_matrix(tokens_samples)
print(onehot_mat)