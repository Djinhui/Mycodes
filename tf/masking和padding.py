'''
“遮盖”是层得知何时应该跳过/忽略序列输入中的某些时间步骤的方式。
有些层是掩码生成者:Embedding 可以通过输入值来生成掩码（如果 mask_zero=True),Masking 层也可以。
有些层是掩码使用者：它们会在其 __call__ 方法中公开 mask 参数。RNN 层就是如此。
在函数式 API 和序列式 API 中，掩码信息会自动传播。
单独使用层时，您可以将 mask 参数手动传递给层。
您可以轻松编写会修改当前掩码的层、生成新掩码的层，或使用与输入关联的掩码的层
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. padding
texts = [
  ["Hello", "world", "!"],
  ["How", "are", "you", "doing", "today"],
  ["The", "weather", "will", "be", "nice", "tomorrow"],
]

texts2num = [
  [71, 1331, 4231]
  [73, 8, 3215, 55, 927],
  [83, 91, 1, 645, 1253, 927],
]

'''
此数据是一个嵌套列表，其中各个样本的长度分别为 3、5 和 6
由于深度学习模型的输入数据必须为单一张量（例如在此例中形状为 (batch_size, 6, vocab_size)），
短于最长条目的样本需要用占位符值进行填充（或者，也可以在填充短样本前截断长样本）
'''
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(texts2num, padding="post")
print(padded_inputs)
'''
[[ 711  632   71    0    0    0]
 [  73    8 3215   55  927    0]
 [  83   91    1  645 1253  927]]
'''

# 2. mask
'''
在 Keras 模型中引入输入掩码有三种方式：

添加一个 keras.layers.Masking 层。
使用 mask_zero=True 配置一个 keras.layers.Embedding 层。
在调用支持 mask 参数的层（如 RNN 层）时，手动传递此参数。
'''
# 2.1
embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
masked_output = embedding(padded_inputs)
print(masked_output._keras_mask)

# 2.2.1
masking_layer = layers.Masking()
# Simulate the embedding lookup by expanding the 2D input to 3D,
# with embedding dimension of 10.
unmasked_embedding = tf.cast(
    tf.tile(tf.expand_dims(padded_inputs, axis=-1), [1, 1, 10]), tf.float32
)

masked_embedding = masking_layer(unmasked_embedding)
print(masked_embedding._keras_mask)

# 2.2.2
model = keras.Sequential(
    [layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True), layers.LSTM(32),]
)

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)(inputs)
outputs = layers.LSTM(32)(x)

model = keras.Model(inputs, outputs)

# 3.
class MyLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.embedding = layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
        self.lstm = layers.LSTM(32)

    def call(self, inputs):
        x = self.embedding(inputs)
        # Note that you could also prepare a `mask` tensor manually.
        # It only needs to be a boolean tensor
        # with the right shape, i.e. (batch_size, timesteps).
        mask = self.embedding.compute_mask(inputs)
        output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
        return output


layer = MyLayer()
x = np.random.random((32, 10)) * 100
x = x.astype("int32")
layer(x)