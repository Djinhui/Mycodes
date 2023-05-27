'''
一个好的嵌入空间会根据周围词的不同而为一个词提供不同的向量表示。
这就是自注意力self-attention的作用。自注意力的目的是利用序列中相关词元的表示来调节某个词元的表示,从而生成上下文感知的词元表示.

你有一个参考序列，用于描述你要查找的内容：查询。
你有一个知识体系，并试图从中提取信息：值。
每个值都有一个键，用于描述这个值，并可以很容易与查询进行对比。你只需将查询与键进行匹配，然后返回值的加权和。


在实践中，键和值通常是同一个序列。
比如在机器翻译中，查询是目标序列，键和值则都是源序列,Decoder-Encoder注意力层中,Queries from Decoder states, Keys and Values from Encoder states
如果你只做序列分类，那么查询、键和值这三者是相同的：将一个序列与自身进行对比，用整个序列的上下文来丰富每个词元的表示
'''
# 伪代码

import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def self_attention(input_sequence):
    output = np.zeros(shape=input_sequence.shape)
    for i, pivot_vector in enumerate(input_sequence): # 对输入序列的每个词元进行迭代(把输入理解为一句话的EmbeddingVector)
        scores = np.zeros(shape=(len(input_sequence),))
        for j, vector in enumerate(input_sequence):
            scores[i] = np.dot(pivot_vector, vector.T) # 计算该词元与其余每个词(包括自身)之间的点积(注意力分数)

        scores /= np.sqrt(input_sequence.shape[1])
        scores = softmax(scores)
        new_pivot_representation = np.zeros(shape=pivot_vector.shape)
        for j, vector in enumerate(input_sequence):
            new_pivot_representation += vector * scores[j] # 利用注意力分数进行加权，对所有词元求和

        output[i] = new_pivot_representation

    return output  # 每个词元的新表示


# Keras内置层MultiHeadAttention
'''
num_heads = 4
embed_dim = 256
mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
outputs = mha_layer(inputs, inputs, inputs)

outputs = sum(values * pairwise_scores(query, keys))
'''

# 一 利用Transformer编码器用于文本分类[还未使用位置编码]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def layer_normalization(batch_of_sequences):  #←----输入形状：(batch_size, sequence_length, embedding_dim)
    mean = np.mean(batch_of_sequences, keepdims=True, axis=-1)  #←---- (本行及以下1行)计算均值和方差，仅在最后一个轴（−1轴）上汇聚数据
    variance = np.var(batch_of_sequences, keepdims=True, axis=-1)
    return (batch_of_sequences - mean) / variance

def batch_normalization(batch_of_images):  #←----输入形状：(batch_size, height, width, channels)
    mean = np.mean(batch_of_images, keepdims=True, axis=(0, 1, 2)) # ←---- (本行及以下1行)在批量轴（0轴）上汇聚数据，这会在一个批量的样本之间形成相互作用
    variance = np.var(batch_of_images, keepdims=True, axis=(0, 1, 2))
    return (batch_of_images - mean) / variance

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(self, **kwargs)
        self.embed_dim = embed_dim 
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation='relu'), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None: # Embedding 层生成的掩码是二维的，但注意力层的输入应该是三维或四维的
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self): # 实现序列化，以便保存模型
        config =  super().get_config()
        config.update({'embed_dim':self.embed_dim, 'num_heads':self.num_heads, 'dense_dim':self.dense_dim})
        return config

'''
config = layer.get_config()
new_layer = layer.__class__.from_config(config)  ←---- config不包含权重值，因此该层的所有权重都是从头初始化的


model = keras.models.load_model(
    filename, custom_objects={"PositionalEmbedding": PositionalEmbedding})
'''  

'''
aclImdb/
...train/
......pos/
......neg/
...test/
......pos/
......neg/
'''

batch_size = 32
train_ds = keras.utils.text_dataset_from_directory("aclImdb/train", batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory("aclImdb/val", batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory("aclImdb/test", batch_size=batch_size)

text_only_train_ds = train_ds.map(lambda x, y: x)  # ←----准备一个数据集，只包含原始文本输入（不包含标签）

max_length = 600
max_tokens = 20000
text_vectorization = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=max_length,  #←----为保持输入大小可控，我们在前600个单词处截断输入。评论的平均长度是233个单词，只有5%的评论超过600个单词
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

vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)  #←---- TransformerEncoder返回的是完整序列，所以我们需要用全局汇聚层将每个序列转换为单个向量，以便进行分类
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("transformer_encoder.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20,
          callbacks=callbacks)

model = keras.models.load_model(
    "transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder})  # ←----在模型加载过程中提供自定义的TransformerEncoder类
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# 二 利用Transformer编码器用于文本分类[使用位置编码]
# 定义一个可学习的位置编码层，非原始正弦编码
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs): # 位置嵌入需要实现直到序列长度
        super().__init__(**kwargs)
        self.token_embedding = layers.Embedding(input_dim=input_dim, output_dim=output_dim) # input_dim=vocab_size
        self.position_embeddings = layers.Embedding(input_dim=sequence_length,output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.out_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embedding(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim":self.out_dim, "sequence_length":self.sequence_length, "input_dim":self.input_dim})
        return config


vocab_size = 20000
sequence_length = 600
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs) #  ←----注意这行代码！
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras",
                                    save_best_only=True)
]
model.fit(int_train_ds, validation_data=int_val_ds, epochs=20,
     callbacks=callbacks)
model = keras.models.load_model(
    "full_transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder,
                    "PositionalEmbedding": PositionalEmbedding})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

