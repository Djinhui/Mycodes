import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb

'''
在tensorflow中完成文本数据预处理的常用方案有两种
第一种是利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Sequence构建文本数据生成器管道。
第二种是使用tf.data.Dataset搭配tf.keras.layers.experimental.preprocessing.TextVectorization预处理层。
'''
# 1. 基于具体任务训练词向量
max_feature=1000 #每条文本只保留最常见的1000个词
max_len=20 #每条文本单词个数最多为20

# x_train 和x_test是代表单词的整数索引
(x_train,y_train),(x_test,t_test)=imdb.load_data(num_words=max_feature)
# 将整数列表转换成形状为(samples,maxlen) 的二维整数张量
x_train = keras.preprocessing.sequence.pad_sequences(x_train, max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, max_len)

e_dim = 8
model = keras.models.Sequential()
model.add(keras.layers.Embedding(max_feature, e_dim, input_length=max_len))
model.add(keras.layers.LSTM(32))
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', metrics=['acccuracy'], optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)


# 2. 使用预训练的词向量
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
# 使用原始imdb数据集(即英文语句序列）
train = pd.read_csv('original_imdb_texts.txt', sep='\t', header=None)
train.columns = ['label', 'text']

text = train.text
label = train.label

# 对文本进行分词
max_len = 100
max_words = 1000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text) # 将字符串转换为单词的整数索引组成的列表 
# one_hot_results=tokenizer.texts_to_matrix(text,mode='binary')
word_index = tokenizer.word_index
word_index_length = len(word_index)
print(f'found {word_index_length} unique tokens')
data = pad_sequences(sequence, max_len)

import numpy as np

indices=np.arange(data.shape[0])

np.random.shuffle(indices)
data=data[indices]
label=label[indices]
x_val=data[:5000]
y_val=label[:5000]
x_train=data[5000:]
y_train=label[5000:]

# 加载GloVe
embedding_index = {}
f = open('./glove.6B.100.txt', 'r', encoding='mac_roman')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.array(values[1:], dtype='float32')
    embedding_index[word] = coefs

f.close()


# 构建可以加载到Embedding层的嵌入矩阵
expanding_dim = 100
embedding_matrx = np.zeros((max_words, expanding_dim)) # 嵌入矩阵(1000*100)
for word, i in word_index.items():
    if i <  max_words:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrx[i] = embedding_vector

model=keras.models.Sequential()
model.add(keras.layers.Embedding(1000, 100, input_length=max_len))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1,activation='sigmoid'))

model.layers[0].set_weights([embedding_matrx])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))



# 可以使用Keras方式保存模型，也可以使用TensorFlow原生方式保存。
# 前者仅仅适合使用Python环境恢复模型，后者则可以跨平台进行模型部署。

# 1. Keras方式
# 保存模型结构及权重
model.save('../../data/keras_model.h5')  
del model  #删除现有模型
# identical to the previous one
model = keras.models.load_model('../../data/keras_model.h5')
model.evaluate(x_train, y_train)
# 保存模型结构
json_str = model.to_json()
# 恢复模型结构
model_json = keras.models.model_from_json(json_str)
#保存模型权重
model.save_weights('../../data/keras_model_weight.h5')
# 恢复模型结构
model_json = keras.models.model_from_json(json_str)
model_json.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
# 加载权重
model_json.load_weights('../../data/keras_model_weight.h5')
model_json.evaluate(x_train,y_train)

# 2. Tensorflow 方式
# 保存权重，该方式仅仅保存权重张量
model.save_weights('../../data/tf_model_weights.ckpt',save_format = "tf")
# 保存模型结构与模型参数到文件,该方式保存的模型具有跨平台性便于部署
model.save('../../data/tf_model_savedmodel', save_format="tf")
print('export saved model.')
model_loaded = tf.keras.models.load_model('../../data/tf_model_savedmodel')
model_loaded.evaluate(x_train,y_train)













