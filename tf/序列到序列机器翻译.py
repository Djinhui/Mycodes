import random
import re
import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

layers.experimental.preprocessing.TextVectorization
# layers.TextVectorization（高版本）


text_file = "spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:  #←----对文件中的每一行进行遍历
    english, spanish = line.split("\t")  #←----每一行都包含一个英语句子和它的西班牙语译文，二者以制表符分隔
    spanish = "[start] " + spanish + " [end]"  #←----将"[start]"和"[end]"分别添加到西班牙语句子的开头和结尾
    text_pairs.append((english, spanish))

# 常见的训练集、验证集和测试集
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

# 将英语和西班牙语的文本对向量化
strip_chars = string.punctuation + "¿"  #←---- (本行及以下6行)为西班牙语的TextVectorization层准备一个自定义的字符串标准化函数：保留"["和"]"，但去掉"¿"（同时去掉string.punctuation中的其他所有字符）
# 保留"["和"]"：将[和]替换为''，TextVectorization就不会再把[和]replace了
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

vocab_size = 15000  # ←---- (本行及以下1行)为简单起见，只查看每种语言前15 000个最常见的单词，并将句子长度限制为20个单词
sequence_length = 20

source_vectorization = layers.TextVectorization(  # ←----英语层 默认标准化处理为：lower+replace(string.punctuation,'')
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = layers.TextVectorization(  #←----西班牙语层
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,  #←----生成的西班牙语句子多了一个词元，因为在训练过程中需要将句子偏移一个时间步
    standardize=custom_standardization,
)
train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)  # ←---- (本行及以下1行)学习每种语言的词表
target_vectorization.adapt(train_spanish_texts)


# 准备翻译任务的数据集
'''
返回一个元组(inputs, target),其中inputs是一个字典,包含两个键,分别是“编码器输入”（英语句子）和“解码器输入”（西班牙语句子）,
target则是向后偏移一个时间步的西班牙语句子
'''
batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],  #←----输入西班牙语句子不包含最后一个词元，以保证输入和目标具有相同的长度
    }, spa[:, 1:])  #←----目标西班牙语句子向后偏移一个时间步。二者长度相同，都是20个单词

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()  #←----利用内存缓存来加快预处理速度

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

'''
>>> for inputs, targets in train_ds.take(1):
>>>     print(f"inputs['english'].shape: {inputs['english'].shape}")
>>>     print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
>>>     print(f"targets.shape: {targets.shape}")
inputs["english"].shape: (64, 20)
inputs["spanish"].shape: (64, 20)
targets.shape: (64, 20)
'''

# 一：RNN的序列到序列学习
'''
有问题
inputs = keras.Input(shape=(sequence_length,), dtype="int64")
x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)
x = layers.LSTM(32, return_sequences=True)(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)
'''

# 二:基于GRU的编码器和解码器
embed_dim = 256
latent_dim = 1024

# 基于GRU的编码器
source = keras.Input(shape=(None,), dtype="int64", name="english")  #←----不要忘记掩码，它对这种方法来说很重要
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)  #←----这是英语源句子。指定输入名称，我们就可以用输入组成的字典来拟合模型
encoded_source = layers.Bidirectional(
    layers.GRU(latent_dim), merge_mode="sum")(x)  #←----编码后的源句子即为双向GRU的最后一个输出

# 基于GRU的解码器与端到端模型
past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")  #←----这是西班牙语目标句子
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)  #←----不要忘记使用掩码
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)  #←----编码后的源句子作为解码器GRU的初始状态
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)  #←----预测下一个词元
seq2seq_rnn = keras.Model([source, past_target], target_next_step)  #←----端到端模型：将源句子和目标句子映射为偏移一个时间步的目标句子

seq2seq_rnn.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

# 利用RNN编码器和RNN解码器来翻译新句子
'''
首先将种子词元"[start]"与编码后的英文源句子一起输入解码器模型。
我们得到下一个词元的预测结果，并不断将其重新输入解码器，每次迭代都采样一个新的目标词元，直到遇到"[end]"或达到句子的最大长度
'''

spa_vocab = target_vectorization.get_vocabulary()  #←---- (本行及以下1行)准备一个字典，将词元索引预测值映射为字符串词元
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"  #←----种子词元
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])
        next_token_predictions = seq2seq_rnn.predict(  #←---- (本行及以下2行)对下一个词元进行采样
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]  #←---- (本行及以下1行)将下一个词元预测值转换为字符串，并添加到生成的句子中
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":  #←----退出条件：达到最大长度或遇到停止词元
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))


# 二：基于Transfomer
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
    

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True  #←----这一属性可以确保该层将输入掩码传递给输出。Keras中的掩码是可选项。如果一个层没有实现compute_mask()并且没有暴露这个supports_masking属性，那么向该层传入掩码则会报错

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
    
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:,tf.newaxis] # (sequence_length, 1)
        j = tf.range(sequence_length) # (sequence_length, )
        mask = tf.cast(i >= j, dtype='int32') # (sequence_length, sequence_length)一个下三角矩阵
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1,1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult) # (batch_size, sequence_length, sequence_length)
    
    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis,:], dtype='int32')
            padding_mask = tf.minimum(padding_mask, causal_mask)
        else:
            padding_mask = causal_mask

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        attention_output_1 = self.layernorm_1(inputs+attention_output_1)
        attention_output_2 = self.attention_2(query=attention_output_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        attention_output_2 = self.layernorm_2(attention_output_1+attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(proj_output+attention_output_2)

# End-to-end
embed_dim = 256
dense_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)  #←----对源句子进行编码

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs) # ←----对目标句子进行编码，并将其与编码后的源句子合并
x = layers.Dropout(0.5)(x)

decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x) # ←----在每个输出位置预测一个单词
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

transformer.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])
transformer.fit(train_ds, epochs=30, validation_data=val_ds)

# 利用Transformer模型来翻译新句子
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization(
            [decoded_sentence])[:, :-1]
        predictions = transformer(
            [tokenized_input_sentence, tokenized_target_sentence])  #←---- (本行及以下1行)对下一个词元进行采样
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]  #←---- (本行及以下1行)将预测的下一个词元转换为字符串，并添加到生成的句子中
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":  #←---- (本行及以下1行)退出条件
            break
    return decoded_sentence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sentence = random.choice(test_eng_texts)
    print("-")
    print(input_sentence)
    print(decode_sequence(input_sentence))