'''
aclImdb/
...train/
......pos/
......neg/
...test/
......pos/
......neg/
'''

import os, pathlib, shutil, random
import tensorflow as tf
from tensorflow import keras

# 准备一个验证集
base_dir = pathlib.Path("aclImdb")
val_dir = base_dir / "val"
train_dir = base_dir / "train"
for category in ("neg", "pos"):
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)  #←----使用种子随机打乱训练文件列表，以确保每次运行代码都会得到相同的验证集
    num_val_samples = int(0.2 * len(files))  #←---- (本行及以下1行)将20%的训练文件用于验证
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,  #←---- (本行及以下1行)将文件移动到aclImdb/val/neg目录和aclImdb/val/pos目录
                    val_dir / category / fname)



batch_size = 32
train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)

'''
>>> for inputs, targets in train_ds:
>>>     print("inputs.shape:", inputs.shape)
>>>     print("inputs.dtype:", inputs.dtype)
>>>     print("targets.shape:", targets.shape)
>>>     print("targets.dtype:", targets.dtype)
>>>     print("inputs[0]:", inputs[0])
>>>     print("targets[0]:", targets[0])
>>>     break
inputs.shape: (32,)
inputs.dtype: <dtype: "string">
targets.shape: (32,)
targets.dtype: <dtype: "int32">
inputs[0]: tf.Tensor(b"This string contains the movie review.", shape=(),
     dtype=string)
targets[0]: tf.Tensor(1, shape=(), dtype=int32)

'''
# 一：词袋模型

# 1. 1-gram
text_vectorization = TextVectorization( 
    max_tokens=20000,
    output_mode="multi_hot",  
)

text_only_train_ds = train_ds.map(lambda x, y: x)  # ←----准备一个数据集，只包含原始文本输入（不包含标签）
text_vectorization.adapt(text_only_train_ds)  # ←----利用adapt()方法对数据集词表建立索引

binary_1gram_train_ds = train_ds.map(  #←---- (本行及以下8行)分别对训练、验证和测试数据集进行处理。一定要指定num_parallel_calls，以便利用多个CPU内核
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_1gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_1gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

'''
>>> for inputs, targets in binary_1gram_train_ds:
>>>     print("inputs.shape:", inputs.shape)
>>>     print("inputs.dtype:", inputs.dtype)
>>>     print("targets.shape:", targets.shape)
>>>     print("targets.dtype:", targets.dtype)
>>>     print("inputs[0]:", inputs[0])
>>>     print("targets[0]:", targets[0])
>>>     break
inputs.shape: (32, 20000)  ←----输入是由20 000维向量组成的批量
inputs.dtype: <dtype: "float32">
targets.shape: (32,)
targets.dtype: <dtype: "int32">
inputs[0]: tf.Tensor([1. 1. 1. ... 0. 0. 0.], shape=(20000,), dtype=float32)  ←----这些向量由0和1组成
targets[0]: tf.Tensor(1, shape=(), dtype=int32)

'''

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_1gram.keras",
                                    save_best_only=True)
]
model.fit(binary_1gram_train_ds.cache(), #  ←---- (本行及以下1行)对数据集调用cache()，将其缓存在内存中：利用这种方法，我们只需在第一轮做一次预处理，在后续轮次可以复用预处理的文本。只有在数据足够小、可以装入内存的情况下，才可以这样做
          validation_data=binary_1gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("binary_1gram.keras")
print(f"Test acc: {model.evaluate(binary_1gram_test_ds)[1]:.3f}")


# 2. 2-grams
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="multi_hot",
)

text_vectorization.adapt(text_only_train_ds)
binary_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
binary_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras",
                                    save_best_only=True)
]
model.fit(binary_2gram_train_ds.cache(),
          validation_data=binary_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("binary_2gram.keras")
print(f"Test acc: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")

# 3. 计算二元语法的出现次数
text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="count"
)

# 4. TF-IDF
import math

def tfidf(term, document, dataset):
    term_freq = document.count(term)
    doc_freq = math.log(sum(doc.count(term) for doc in dataset) + 1)
    return term_freq / doc_freq


text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="tf_idf",
)

text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
tfidf_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
tfidf_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("tfidf_2gram.keras",
                                    save_best_only=True)
]
model.fit(tfidf_2gram_train_ds.cache(),
          validation_data=tfidf_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("tfidf_2gram.keras")
print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

'''
导出能够处理原始字符串的模型
inputs = keras.Input(shape=(1,), dtype="string")  ←----每个输入样本都是一个字符串
processed_inputs = text_vectorization(inputs)  ←----应用文本预处理
outputs = model(processed_inputs)  ←----应用前面训练好的模型
inference_model = keras.Model(inputs, outputs)  ←----将端到端的模型实例化

import tensorflow as tf
raw_text_data = tf.convert_to_tensor([["That was an excellent movie, I loved it."],])
predictions = inference_model(raw_text_data)
print(f"{float(predictions[0] * 100):.2f} percent positive")

'''

# 二：序列模型
from tensorflow.keras import layers

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,  #←----为保持输入大小可控，我们在前600个单词处截断输入。这是一个合理的选择，因为评论的平均长度是233个单词，只有5%的评论超过600个单词
)
text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)
int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4)

# 1. 构建于one-hot编码的向量序列之上的序列模型
# Do not do this , 非常慢且不合理

import tensorflow as tf
inputs = keras.Input(shape=(None,), dtype="int64") # ←----每个输入是一个整数序列

# 每个输入样本被编码成尺寸为(600, 20000)的矩阵（每个样本包含600个单词，共有20 000个可能的单词）
embedded = tf.one_hot(inputs, depth=max_tokens) # ←----将整数编码为20 000维的二进制向量
x = layers.Bidirectional(layers.LSTM(32))(embedded)  #←----添加一个双向LSTM
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)  #←----最后添加一个分类层
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("one_hot_bidir_lstm.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

# 2. 利用Embedding层学习词嵌入
# embedding_layer = layers.Embedding(input_dim=max_tokens, output_dim=256)  #←---- Embedding层至少需要两个参数：词元个数和嵌入维度（这里是256）
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("embeddings_bidir_gru.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

'''
填充和掩码

输入序列中包含许多0。这是因为我们在TextVectorization层中使用了output_sequence_length=max_length选项（max_length为600），
也就是说，多于600个词元的句子将被截断为600个词元，而少于600个词元的句子则会在末尾用0填充，使其能够与其他序列连接在一起，形成连续的批量。
我们使用的是双向RNN，即两个RNN层并行运行，一个正序处理词元，另一个逆序处理相同的词元。
按正序处理词元的RNN，在最后的迭代中只会看到表示填充的向量。如果原始句子很短，那么这可能包含几百次迭代。
在读取这些无意义的输入时，存储在RNN内部状态中的信息将逐渐消失。我们需要用某种方式来告诉RNN，它应该跳过这些迭代。
有一个API可以实现此功能：掩码（masking）

向Embedding层传入mask_zero=True来启用它。你可以用compute_mask()方法来获取掩码

Embedding层能够生成与输入数据相对应的掩码。这个掩码是由1和0（或布尔值True/False）组成的张量，形状为(batch_size, sequence_length)，
其元素mask[i, t]表示第i个样本的第t个时间步是否应该被跳过（如果mask[i,t]为0或False，则跳过该时间步，反之则处理该时间步）

Keras会将掩码自动传递给能够处理掩码的每一层（作为元数据附加到所对应的序列中）。RNN层会利用掩码来跳过被掩码的时间步
>>> embedding_layer = layers.Embedding(input_dim=10, output_dim=256, mask_zero=True)
>>> some_input = [
... [4, 3, 2, 1, 0, 0, 0],
... [5, 4, 3, 2, 1, 0, 0],
... [2, 1, 0, 0, 0, 0, 0]]
>>> mask = embedding_layer.compute_mask(some_input)
<tf.Tensor: shape=(3, 7), dtype=bool, numpy=
array([[ True,  True,  True,  True, False, False, False],
       [ True,  True,  True,  True,  True, False, False],
       [ True,  True, False, False, False, False, False]])>

'''

# 3.使用带有掩码的Embedding层
inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

# 4. 使用预训练词嵌入

import numpy as np

# 解析GloVe词嵌入文件
path_to_glove_file = 'glove.6B.100d.txt'
embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

embedding_dim = 100
vocabulary = text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))
embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word, None)

    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = layers.Embedding(max_tokens, embedding_dim,embeddings_initializer=keras.initializers.Constant(embedding_matrix),\
    trainable=False, mask_zero=True)

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("glove_embeddings_sequence_model.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=10,
          callbacks=callbacks)
model = keras.models.load_model("glove_embeddings_sequence_model.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")
