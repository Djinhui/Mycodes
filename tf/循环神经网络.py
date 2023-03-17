'''
简单来说,RNN层会使用for循环对序列的时间步骤进行迭代,同时维持一个内部状态,对截至目前所看到的时间步骤信息进行编码
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128))

# Add a Dense layer with 10 units.
model.add(layers.Dense(10))

model.summary()

# 1. 输出和状态
'''
默认情况下,RNN 层的输出为每个样本包含一个向量。此向量是与最后一个时间步骤相对应的 RNN 单元输出，
包含关于整个输入序列的信息。此输出的形状为 (batch_size, units)，
其中 units 对应于传递给层构造函数的 units 参数。

如果您设置了 return_sequences=True,RNN 层还能返回每个样本的
整个输出序列（每个样本的每个时间步骤一个向量）。此输出的形状为 (batch_size, timesteps, units)。
'''
model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(layers.SimpleRNN(128))

model.add(layers.Dense(10))

model.summary()

# 1.1
'''
此外RNN 层还可以返回其最终内部状态。返回的状态可用于稍后恢复 RNN 执行，或初始化另一个 RNN
此设置常用于编码器-解码器序列到序列模型，其中编码器的最终状态被用作解码器的初始状态。

要配置 RNN 层以返回其内部状态，请在创建该层时将 return_state 参数设置为 True。
请注意LSTM 具有两个状态张量，但 GRU 只有一个。

要配置该层的初始状态，只需额外使用关键字参数 initial_state 调用该层。
请注意，状态的形状需要匹配该层的单元大小，如下例所示。
'''
encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
    encoder_input
)

# Return states in addition to output
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
    encoder_embedded
)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embedded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)

# Pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(64, name="decoder")(
    decoder_embedded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
model.summary()

# 2. RNN 层和 RNN 单元
'''
除内置 RNN 层外,RNN API 还提供单元级 API。与处理整批输入序列的 RNN 层不同,RNN 单元仅处理单个时间步骤。
单元位于 RNN 层的 for 循环内。将单元封装在 keras.layers.RNN 层内，
您会得到一个能够处理序列批次的层，如 RNN(LSTMCell(10))。
从数学上看,RNN(LSTMCell(10)) 会产生和 LSTM(10) 相同的结果。

共有三种内置 RNN 单元，每种单元对应于匹配的 RNN 层。

keras.layers.SimpleRNNCell 对应于 SimpleRNN 层。
keras.layers.GRUCell 对应于 GRU 层。
keras.layers.LSTMCell 对应于 LSTM 层。
'''

# 3. 跨批次有状态性
'''
通常情况下，每次看到新批次时，都会重置 RNN 层的内部状态（即，假定该层看到的每个样本都独立于过去）。
该层将仅在处理给定样本时保持状态。

但如果您的序列非常长，一种有效做法是将它们拆分成较短的序列，
然后将这些较短序列按顺序馈送给 RNN 层，而无需重置该层的状态。

通过在构造函数中设置 stateful=True 来执行上述操作

'''

# 4. 双向RNN
'''
默认情况下,Bidirectional RNN 的输出将是前向层输出和后向层输出的串联。
如果您需要串联等其他合并行为，请更改 Bidirectional 封装容器构造函数中的 merge_mode 参数
'''
model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
) # Output Shape  (None, 5, 128) 
model.add(layers.Bidirectional(layers.LSTM(32))) # Output Shape  (None, 64)
model.add(layers.Dense(10))

model.summary()