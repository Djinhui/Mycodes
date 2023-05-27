'''
首先，将文本标准化，使其更容易处理，比如转换为小写字母或删除标点符号。
然后,将文本拆分为单元［称为词元（token）］,比如字符、单词或词组。这一步叫作词元化。
最后，将每个词元转换为一个数值向量。这通常需要首先对数据中的所有词元建立索引
'''

# 1. 纯python实现
import string
import re
class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {'':0, '[UNK]':1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
                
        self.inverse_vocabulary = dict((v, k) for k, v in self.vocabulary.items())

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return ' '.join(self.inverse_vocabulary.get(i, '[UNK]') for i in int_sequence)


vectorizer = Vectorizer()
dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]
vectorizer.make_vocabulary(dataset)

# 2. 
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


# 3. TextVectorization层
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(
    output_mode="int",  # ←----设置该层的返回值是编码为整数索引的单词序列。还有其他几种可用的输出模式，稍后会看到其效果
)

# 自定义函数的作用对象应该是tf.string张量
def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(lowercase_string, f"[{re.escape(string.punctuation)}]")

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)

text_vectorization = TextVectorization(output_mode='int', standardize=custom_standardization_fn, split=custom_split_fn)

dataset = [
    "I write, erase, rewrite",
    "Erase again, and then",
    "A poppy blooms.",
]

text_vectorization.adapt(dataset)
vocabulary = text_vectorization.get_vocabulary()

test_sentence = "I write, rewrite, and still rewrite again"
encoded_sentence = text_vectorization(test_sentence)
inverse_vocab = dict(enumerate(vocabulary))
decoded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)

# 在tf.data管道中使用TextVectorization层或者将TextVectorization层作为模型的一部分
## 1. 在tf.data管道中使用(推荐在GPU/TPU上使用)
string_dataset = 'StringDataset'
int_sequence_dataset = string_dataset.map(  #←---- string_dataset是一个能够生成字符串张量的数据集
    text_vectorization,
    num_parallel_calls=4)  # ←---- num_parallel_calls参数的作用是在多个CPU内核中并行调用map()

## 2. 第二种用法是将其作为模型的一部分（毕竟它是一个Keras层）
text_input = keras.Input(shape=(), dtype="string") # ←----创建输入的符号张量，数据类型为字符串
vectorized_text = text_vectorization(text_input) # ←----对输入应用文本向量化层
embedded_input = keras.layers.Embedding(...)(vectorized_text)  # ←---- (本行及以下2行)你可以继续添加新层，就像普通的函数式API模型一样
output = ...
model = keras.Model(text_input, output)