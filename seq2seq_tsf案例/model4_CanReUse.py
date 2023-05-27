# https://blog.csdn.net/qq_42189083/article/details/89356188
# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
'''
输入序列为随机产生的整数序列，目标序列是对输入序列前三个元素进行反转后的序列
输入序列		                目标序列
[13, 28, 18, 7, 9, 5]		[18, 28, 13]
[29, 44, 38, 15, 26, 22]	[38, 44, 29]
[27, 40, 31, 29, 32, 1]		[31, 40, 27]
'''

import numpy as np
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model


def generate_sequence(length, n_unique):
    return [random.randint(1, n_unique-1) for _ in range(length)]  # random.randint(a,b)全闭， np.random.randint(a,b)左闭右开

def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        source = generate_sequence(n_in, cardinality)
        taregt = source[:n_out]
        taregt.reverse()

        # 向前偏移一个时间步目标序列
        target_in = [0] + taregt[:-1]

        # 使用onehot编码
        src_encoded = to_categorical(source, num_classes=cardinality)
        tar_encoded = to_categorical(taregt, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)

        X1.append(src_encoded) # Encoder输入
        X2.append(tar2_encoded) # Dencoder输入
        y.append(tar_encoded)   # Decoder目标输出

    return np.array(X1), np.array(X2), np.array(y)


def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]



def define_model(n_input, n_output, n_units):
    # n_input=n_output=n_features

    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c] # 保留编码状态

    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) # 将编码器状态作为解码器初始状态
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units, ))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # 输入序列编码得到状态
    state = infenc.predict(source)
    # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1,1, cardinality) 
    output = []
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        output.append(yhat[0, 0, :])
        state = [h, c]
        target_seq = yhat
    return np.array(output)

n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
print(X1.shape) # (1, 6, 51)
print(X2.shape) # (1, 3, 51)
print(y.shape)  # (1, 3, 51)
print('X1=%s, X2=%s, y=%s' % (one_hot_decode(X1[0]), one_hot_decode(X2[0]), one_hot_decode(y[0])))
# X1=[32, 16, 12, 34, 25, 24], X2=[0, 12, 16], y=[12, 16, 32]

train_model, infenc, infdec = define_model(n_features, n_features, 128)
train_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# 训练模型
train_model.fit([X1, X2], y, epochs=1)


total, correct = 100, 0
for _ in range(total):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    if np.array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
        correct += 1

print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# 查看预测结果
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
