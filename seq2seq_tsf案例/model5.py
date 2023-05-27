# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras
import numpy as np
from random import randint
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, constraints, initializers, activations
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential, Model
# copy from the file custom_recurrents.py in GitHub project called “keras-attention“.
# https://github.com/datalogue/keras-attention
from attention_decoder import AttentionDecoder  

# attention_decoder.py
def generate_sequence(length, n_unique):
    return [randint(0, n_unique-1) for _ in range(length)]

def one_hot_encode(sequence, n_unique):
    encoding = []
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return np.array(encoding)

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

# prepare data for the LSTM
def get_pair(n_in, n_out, n_unique):
    sequence_in = generate_sequence(n_in, n_unique)
    sequence_out = sequence_in[:n_out] + [0 for _ in range(n_in - n_out)]
    X = one_hot_encode(sequence_in, n_unique)
    y = one_hot_encode(sequence_out, n_unique)
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y

# generate random sequence
sequence = generate_sequence(5, 50)
print(sequence) # [3,18,32,11,36]
# one hot encode
encoded = one_hot_encode(sequence, 50)
print(encoded) #(5,50)
# decode
decoded = one_hot_decode(encoded)
print(decoded) # [3,18,32,11,36]

# generate random sequence
X, y = get_pair(5, 2, 50)
print(X.shape, y.shape) # (1, 5, 50) (1, 5, 50)
print('X=%s, y=%s' % (one_hot_decode(X[0]), one_hot_decode(y[0])))
# X=[12, 20, 36, 40, 12], y=[12, 20, 0, 0, 0]


# ====================Encoder-Decoder Without Attention====================
# just like KP_load

n_features = 50
n_timesteps_in = 5
n_timesteps_out = 2

model = Sequential()
model.add(layers.LSTM(units=150, input_shape=(n_timesteps_in, n_features)))
model.add(layers.RepeatVector(n_timesteps_in)) # 后面补了n_timesteps_in - n_timesteps_out个0，所以target序列长度也是n_timesteps_in
model.add(layers.LSTM(units=150, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(n_features, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(50000):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    model.fit(X, y, epochs=1, verbose=2)

total, correct = 100, 0
for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    if np.array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

for _ in range(10):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))




# ====================Encoder-Decoder With Attention================
model = Sequential()
model.add(layers.LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
model.add(AttentionDecoder(150, n_features))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(50000):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    model.fit(X, y, epochs=1, verbose=2)

total, correct = 100, 0
for _ in range(total):
    X, y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    if np.array_equal(one_hot_decode(y[0]), one_hot_decode(yhat[0])):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

for _ in range(10):
    X,y = get_pair(n_timesteps_in, n_timesteps_out, n_features)
    yhat = model.predict(X, verbose=0)
    print('Expected:', one_hot_decode(y[0]), 'Predicted', one_hot_decode(yhat[0]))