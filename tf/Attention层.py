'''
https://keras.io/api/layers/attention_layers/
'''
# 一：
# tf.keras.layers.Attention(use_scale=False, score_mode="dot", **kwargs), a.k.a. Luong-style attention.

'''
call Arguments:
use_scale: If True, will create a scalar variable to scale the attention scores.
dropout: Float between 0 and 1. 
score_mode: Function to use to compute attention scores, one of {"dot", "concat"}.
inputs: List of the following tensors: 
    * query: Query Tensor of shape [batch_size, Tq, dim]. 
    * value: Value Tensor of shape [batch_size, Tv, dim]. 
    * key: Optional key Tensor of shape [batch_size, Tv, dim]. If not given, will use value for both key and value, which is the most common case.

# The calculation follows the steps:
scores = tf.matmul(query, key, transpose_b=True). # with shape [batch_size, Tq, Tv] 
distribution = tf.nn.softmax(scores). # with shape [batch_size, Tq, Tv]
outputs = return tf.matmul(distribution, value). # with shape [batch_size, Tq, dim]

mask:List of the following tensors:
    * query_mask: A boolean mask Tensor of shape [batch_size, Tq]. If given, the output will be zero at the positions where mask==False. 
    * value_mask: A boolean mask Tensor of shape [batch_size, Tv]. If given, will apply the mask such that values at positions where mask==False do not contribute to the result. 
return_attention_scores: bool, it True, returns the attention scores (after masking and softmax) as an additional output argument. 
training: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout).
use_causal_mask: Boolean. Set to True for decoder self-attention. Adds a mask such that position i cannot attend to positions j > i. This prevents the flow of information from the future towards the past. Defaults to False.

output:
Attention outputs of shape [batch_size, Tq, dim]. 
[Optional] Attention scores after masking and softmax with shape [batch_size, Tq, Tv]

The meaning of query, value and key depend on the application. 
In the case of text similarity, for example, query is the sequence embeddings of the first piece of text and 
value is the sequence embeddings of the second piece of text. key is usually the same tensor as value.
'''

import tensorflow as tf
from tensorflow.python import keras

# Variable-length int sequences.
query_input = tf.keras.Input(shape=(None,), dtype='int32')
value_input = tf.keras.Input(shape=(None,), dtype='int32')

# Embedding lookup.
token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

# Add DNN layers, and create Model.
# ...


# 二：
# tf.keras.layers.AdditiveAttention(use_scale=True, **kwargs) a.k.a. Bahdanau-style attention

'''
call Arguments:
use_scale: If True, will create a scalar variable to scale the attention scores.
dropout: Float between 0 and 1. 
score_mode: Function to use to compute attention scores, one of {"dot", "concat"}.
inputs: List of the following tensors: 
    * query: Query Tensor of shape [batch_size, Tq, dim]. 
    * value: Value Tensor of shape [batch_size, Tv, dim]. 
    * key: Optional key Tensor of shape [batch_size, Tv, dim]. If not given, will use value for both key and value, which is the most common case.

# The calculation follows the steps:
Reshape query and key into shapes [batch_size, Tq, 1, dim] and [batch_size, 1, Tv, dim]
scores = tf.reduce_sum(tf.tanh(query + key), axis=-1). # with shape [batch_size, Tq, Tv] 
distribution = tf.nn.softmax(scores). # with shape [batch_size, Tq, Tv]
outputs = return tf.matmul(distribution, value). # with shape [batch_size, Tq, dim]

mask:List of the following tensors:
    * query_mask: A boolean mask Tensor of shape [batch_size, Tq]. If given, the output will be zero at the positions where mask==False. 
    * value_mask: A boolean mask Tensor of shape [batch_size, Tv]. If given, will apply the mask such that values at positions where mask==False do not contribute to the result. 
return_attention_scores: bool, it True, returns the attention scores (after masking and softmax) as an additional output argument. 
training: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout).
use_causal_mask: Boolean. Set to True for decoder self-attention. Adds a mask such that position i cannot attend to positions j > i. This prevents the flow of information from the future towards the past. Defaults to False.

output:
Attention outputs of shape [batch_size, Tq, dim]. 
[Optional] Attention scores after masking and softmax with shape [batch_size, Tq, Tv]

The meaning of query, value and key depend on the application. 
In the case of text similarity, for example, query is the sequence embeddings of the first piece of text and 
value is the sequence embeddings of the second piece of text. key is usually the same tensor as value.
'''

# Variable-length int sequences.
query_input = tf.keras.Input(shape=(None,), dtype='int32')
value_input = tf.keras.Input(shape=(None,), dtype='int32')

# Embedding lookup.
max_tokens = 1000
dimension = 64
token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.AdditiveAttention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

# Add DNN layers, and create Model.
# ...

# 三：
'''
tf.keras.layers.MultiHeadAttention(
    num_heads, <--  Number of attention heads.
    key_dim,  <-- Size of each attention head for query and key.
    value_dim=None, <-- Size of each attention head for value.
    dropout=0.0,
    use_bias=True, <-- Boolean, whether the dense layers use bias vectors/matrices.
    output_shape=None, <-- The expected shape of an output tensor, besides the batch and sequence dims. If not specified, projects back to the key feature dim.
    attention_axes=None, <-- axes over which the attention is applied. None means attention over all axes, but batch, heads, and features.
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs
)

Call arguments

query: Query Tensor of shape (B, T, dim).
value: Value Tensor of shape (B, S, dim).
key: Optional key Tensor of shape (B, S, dim). If not given, will use value for both key and value, which is the most common case.
attention_mask: a boolean mask of shape (B, T, S), that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.
return_attention_scores: A boolean to indicate whether the output should be (attention_output, attention_scores) if True, or attention_output if False. Defaults to False.
training: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout). Defaults to either using the training mode of the parent layer/model, or False (inference) if there is no parent layer.
use_causal_mask: A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens (e.g., used in a decoder Transformer).


Returns

attention_output: The result of the computation, of shape (B, T, E), where T is for target sequence shapes and E is the query input last dimension if output_shape is None. Otherwise, the multi-head outputs are projected to the shape specified by output_shape.
attention_scores: [Optional] multi-head attention coefficients over attention axes. [batch_size, num_heads, Tq, Tv]
'''

'''
Performs 1D cross-attention over two sequence inputs with an attention mask. Returns the additional attention weights over heads.
>>> layer = MultiHeadAttention(num_heads=2, key_dim=2)
>>> target = tf.keras.Input(shape=[8, 16])
>>> source = tf.keras.Input(shape=[4, 16])
>>> output_tensor, weights = layer(target, source,return_attention_scores=True)
>>> print(output_tensor.shape)
(None, 8, 16)
>>> print(weights.shape)
(None, 2, 8, 4)


Performs 2D self-attention over a 5D input tensor on axes 2 and 3.

>>> layer = MultiHeadAttention(num_heads=2, key_dim=2, attention_axes=(2, 3))
>>> input_tensor = tf.keras.Input(shape=[5, 3, 4, 16])
>>> output_tensor = layer(input_tensor, input_tensor)
>>> print(output_tensor.shape)
(None, 5, 3, 4, 16)


'''