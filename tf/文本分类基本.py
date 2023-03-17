import tensorflow as tf
from tensorflow import keras
import numpy as np

'''
在tensorflow中完成文本数据预处理的常用方案有两种
第一种是利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Sequence构建文本数据生成器管道。
第二种是使用tf.data.Dataset搭配tf.keras.layers.experimental.preprocessing.TextVectorization预处理层。
'''

'''1. 使用词向量'''
imdb = keras.datasets.imdb

# 参数 num_words=10000 保留了训练数据中最常出现的 10,000 个单词
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# 评论文本被转换为整数值，其中每个整数代表词典中的一个单词
print(train_data[0]) # [1, 14, 22, 16, ..., 167, 2, 336]

# 电影评论可能具有不同的长度
len(train_data[0]), len(train_data[1]) # (218, 189)

# 将整数转换回单词

word_index = imdb.get_word_index() # 一个映射单词到整数索引的词典

# 保留第一个索引
word_index = {k:(v+3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2 # known
word_index['<UNUSED>'] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

decode_review(train_data[0]) # "<START> this film was just brilliant ... "

# 由于电影评论长度必须相同，我们将使用 pad_sequences 函数来使长度标准化
# 电影评论长度不足256向后填充0
train_data = keras.preprocessing.sequence.pad_sequences(train_data, maxlen=256, value=word_index['<PAD>'], padding='post')
test_data = keras.preprocessing.sequence.pad_sequences(test_data, maxlen=256, value=word_index['<PAD>'], padding='post')
len(train_data[0]), len(train_data[1]) # (256, 256)

# 输入形状是用于电影评论的词汇数目（10,000 词）
vocab_size = 10000
model = keras.Sequential()
'''
Embedding层:这里input_legth = maxlen=256, output_dim=16=嵌入维度
Input shape:
  2D tensor with shape: (batch_size, input_length).

Output shape:
  3D tensor with shape: (batch_size, input_length, output_dim)
'''
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data,  test_labels, verbose=2)

print(results)

history_dict = history.history
history_dict.keys() # dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])


import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


'''2. 使用文本嵌入向量'''
import tensorflow_hub as hub
import tensorflow_datasets as tfds
# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch # <tf.Tensor: shape=(10,), dtype=string, numpy=array([b'This was an absolutely...', b'The film is based..',b'',...b''])
train_labels_batch # <tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])>

# 使用来自 TensorFlow Hub 的 预训练文本嵌入向量模型，名称为 google/nnlm-en-dim50/2
# 请注意无论输入文本的长度如何，嵌入（embeddings）输出的形状都是：(num_examples, embedding_dimension)
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3]) # <tf.Tensor: shape=(3, 50), dtype=float32, numpy=array([[...]])

model = tf.keras.Sequential()
model.add(hub_layer)  # Output Shape (None, 16)
model.add(tf.keras.layers.Dense(16, activation='relu'))  # Output Shape (None, 16)
model.add(tf.keras.layers.Dense(1))  # Output Shape (None, 1)
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))