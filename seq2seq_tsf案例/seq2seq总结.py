import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import dot, Activation, concatenate
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Attention, GRU, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib


lookback = 96
delay = 24
indim=1
outdim=1

'''
Keras:LSTM的两个参数:(torch LSTM还要考虑num_layers和num_directions参数)
return_sequences=False, return_state=False: return the last hidden state: state_h
return_sequences=True, return_state=False: return stacked hidden states (num_timesteps * num_cells): one hidden state output for each input time step
return_sequences=False, return_state=True: return 3 arrays: state_h, state_h, state_c
return_sequences=True, return_state=True: return 3 arrays: stacked hidden states, last state_h, last state_c
'''



# M1.=============================KP_Load=====================================
'''
将encoder的输出的last_hidden_h(return_sequences=False)RepeatVector(delay)次作为decoder的输入
dencoder的初始状态采用随机化; decoder输出的stack_hidden_h(return_sequences=True)经TimeDistributed(Dense(target输出维度outdim))转换为目标输出
'''
# inputs:(Batchsize, 96, 1)
# targets:(Batchsize, 24, 1)

def lstm_model():
    model = Sequential()
    model.add(LSTM(32,return_sequences=True, input_shape=(96, 1)))
    model.add(LSTM(32))
    model.add(RepeatVector(24))
    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    opt = Adam(lr=0.001) 
    model.compile(optimizer=opt, loss='mae')
    print(model.summary())
    return model


def train_model():
    model = lstm_model()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min', factor=0.5, min_delta=0.001)
    es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=10)
    date2str = datetime.now().strftime('%Y%m%d %H:%M:%S')
    date2str = date2str.replace(' ','').replace(':','')
    ckpt = ModelCheckpoint('batchsize_{}.h5'.format(32),monitor='val_loss',
                          verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
    history = model.fit_generator('train_dataset', steps_per_epoch=100, epochs=100, validation_data='val_dataset',
                                 validation_steps=10, callbacks=[ckpt,es])
    
    return model, history



# M2.===========================================model1.py==============================
# https://blog.csdn.net/zengNLP/article/details/124375813?spm=1001.2014.3001.5502
# 结合Attention
# encoder需要输出stack_hidden_h(return_sequence=True)用于计算attention_weight， 同时将encoder的[last_hidden_h,last_hidden_c](return_state=True)作为decoder的初始状态
class Encoder(Model):
    def __init__(self, hidden_units):
        super(Encoder, self).__init__()
        self.encoder_gru = GRU(hidden_units, return_sequences=True, return_state=True, name='encoder_gru')
        self.dropout = Dropout(rate=0.5)

    def call(self, inputs):
        encoder_outputs, state_h = self.encoder_gru(inputs) # gru没有cell state
        return encoder_outputs, state_h
    
# deocder需要输出stack_hidden_h(return_sequence=True)用于计算attention_weight,同时接受encoder的[last_hidden_h,last_hidden_c]作为decoder的初始状态
class Decoder(Model):
    def __init__(self, hidden_units):
        super(Decoder, self).__init__()
        self.decoder_gru = GRU(hidden_units,return_sequences=True, return_state=True, name='decoder_gru')
        self.attention = Attention()
        self.dropout = Dropout(rate=0.5)

    def call(self, enc_outputs, dec_inputs, state_inputs):
        dec_outputs, dec_state_h = self.decoder_gru(dec_inputs, initial_state=state_inputs)
        attention_output = self.attention([dec_outputs, enc_outputs])
        return attention_output, dec_state_h
    

loss_fn = tf.keras.losses.MeanAbsoluteError()
def seq2seq_attention(encode_shape, decode_shape, hidden_units, output_dim):
    encoder_inputs = Input(shape=encode_shape, name='encoder_input')
    # decoder_inputs可以时enc_state_h的Repeat(如1.KP_Load)， 或着target的shifted left 1步
    # eg. encoder_input(1,2,3,4,5,6), decoder_target(7,8,9), decoder_input(<sos>,7,8) ??? 自己猜测

    decoder_inputs = Input(shape=decode_shape, name='decoder_input') 
    encoder = Encoder(hidden_units)
    enc_outputs, enc_state_h = encoder(encoder_inputs)
    dec_states_inputs = enc_state_h

    decoder = Decoder(hidden_units)
    attention_output, dec_state_h = decoder(enc_outputs, decoder_inputs, dec_states_inputs)

    # 将经Attention加权的context输入Dense
    dense_output = Dense(output_dim, activation='sigmoid', name='dense')(Dropout(rate=0.5)(attention_output))

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_output)
    model.summary()

    opt = Adam()
    model.compile(loss=loss_fn, optimizer=opt)

    return model

model = seq2seq_attention((96,1), (24, 1), 50, 1)


# M3.======================================model2.py============================
# https://blog.csdn.net/u010329292/article/details/129450576
# 采用GRUCell的方式，训练也是一步一步的训练


# M4.=================================model3_1.py & model3_2.py=========================
# https://blog.csdn.net/Cyril_KI/article/details/126563659
# https://blog.csdn.net/Cyril_KI/article/details/125095225

# Encoder输出last_hidden_h和last_hidden_c作为Decoder的初始状态
class Encoder(Model):
    def __init__(self, args):
        super().__init__()
        self.lstm = LSTM(units=args.hidden_size, input_shape=(args.seq_len, args.input_size),
                                activation='tanh', return_sequences=True, return_state=True) #  return_sequences=False亦可
    def call(self, input_seq):
        output, h, c = self.lstm(input_seq)
        return h, c
    
# Decoder一步一步输入DoForLoop:（batch_size,1，nfeatures）
# Decoder接受Encoder输出的last_hidden_h和last_hidden_c作为Decoder的初始状态
# Decoder返回stack_hidden_h, last_hidden_h, last_hidden_c， stack_hidden_h经Dense()转换为目标输出, last_hidden_h, last_hidden_c动态更新作为下一步状态
class Decoder(Model):
    def __init__(self, args):
        self.lstm = LSTM(units=args.hidden_size, input_shape=(args.seq_len, args.input_size),
                                activation='tanh', return_sequences=True, return_state=True)
        self.fc1 = Dense(64, activation='relu')
        self.fc2 = Dense(args.output_size)

    def call(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        batch_size, input_size = input_seq.shape[0], input_seq.shape[1]
        input_seq = tf.reshape(input_seq, [batch_size, 1, input_size])
        output, h, c = self.lstm(input_seq, initial_state=[h, c])
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.fc1(output)  # pred(batch_size, 1, output_size)
        pred = self.fc2(pred)
        pred = pred[:, -1, :]

        return pred, h, c


class Seq2Seq(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.Encoder = Encoder(args)
        self.Decoder = Decoder(args)

    def call(self, input_seq):
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        res = None
        for t in range(seq_len):
            _input = input_seq[:, t, :]
            output, h, c = self.Decoder(_input, h, c)
            res = output

        return res
    

# M5.============================model4_CanReUse.py============================
# https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
# https://blog.csdn.net/qq_42189083/article/details/89356188

# Encoder输出last_hidden_h和last_hidden_c作为Decoder的初始状态
# Decoder接受Encoder输出的last_hidden_h和last_hidden_c作为Decoder的初始状态
# Decoder返回stack_hidden_h, last_hidden_h, last_hidden_c， stack_hidden_h经Dense()转换为目标输出
# 推断时动态解码，更新hidden_h和hidden_c

'''
输入序列为随机产生的整数序列，目标序列是对输入序列前三个元素进行反转后的序列
encoder输入序列encoder_inputs		    decoder目标序列   decoder输入序列decoder_inputs
[13, 28, 18, 7, 9, 5]		           [18, 28, 13]      [0, 28, 13]
[29, 44, 38, 15, 26, 22]	           [38, 44, 29]      [0, 44, 29]
[27, 40, 31, 29, 32, 1]		           [31, 40, 27]      [0, 40, 27]

数据经onehot编码输入Encoder-Decoder
(Batchsize, 6, MaxNum+1) +1是因为有个起始标志0,序列数字范围[1, MaxNum]
(Batchsize, 3, MaxNum+1) 
(Batchsize, 3, MaxNum+1)
'''

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


# M6.===================================model5.py==================================================
# https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras


# n_features = 50
# n_timesteps_in = 5
# n_timesteps_out = 2

# https://github.com/datalogue/keras-attention
# from attention_decoder import AttentionDecoder 
# model = Sequential()
# model.add(LSTM(150, input_shape=(n_timesteps_in, n_features), return_sequences=True))
# model.add(AttentionDecoder(150, n_features))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

# # Without Attention
# model = Sequential()
# model.add(LSTM(units=150, input_shape=(n_timesteps_in, n_features)))
# model.add(RepeatVector(n_timesteps_in)) # 后面补了n_timesteps_in - n_timesteps_out个0，所以target序列长度也是n_timesteps_in
# model.add(LSTM(units=150, return_sequences=True))
# model.add(TimeDistributed(Dense(n_features, activation='softmax')))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])














# M7. ============!!!!!!!!!!!!!!!!!!!!!!!model6VIPPP.py=================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb

# 1.SimpleSeq2SeqLSTMModel 无Attention

'''
                      ➡------>--RepeatVector(20)---⬇
|--------|            ⬆                       |-------|
|Encoder |-->[last_hidden_h, last_hidden_c]-->|Decoder|-->stack_hidden_h--->TimeDistributed(Dense)
|--------|                                    |-------|

'''
# X_input_train.shape, X_output_train.shape  
# (600, 200, 2),        (600, 20, 2)
'''
encoder lstm generate last state c and hidden state h
use the last cell state c and hidden state h from the encoder as the initial states of the decoder LSTM cell.
The last hidden state of encoder is also copied 20 times, and each copy is input into the decoder LSTM cell together with previous cell state c and hidden state h. 
The decoder outputs hidden state for all the 20 time steps, and these hidden states are connected to a dense layer to output the final result.
'''
n_hidden = 100
input_train = Input(shape=(200, 2)) 
output_train = Input(shape=(20, 2))

## 编码器输出 last state_h and last state_c
# For simple Seq2Seq model, we only need last state_h and last state_c.

# encoder_last_h1 和encoder_last_h2相同
encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
                                                        return_sequences=False, return_state=True)(input_train)
print(encoder_last_h1) #  shape(None,  100)
print(encoder_last_h2) # shape(None,  100)
print(encoder_last_c) # shape(None, 100)

encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

## 解码器接受Encoder输出的last_hidden_h和last_hidden_c作为Decoder的初始状态
## 同时将encoder的输出的last_hidden_h(return_sequences=False)RepeatVector(20)次作为decoder的输入
# make 20 copies of the last hidden state of encoder and use them as input to the decoder.
# The last cell state and the last hidden state of the encoder are also used as the initial states of decoder.
decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
decoder = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, return_state=False,
               return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
print(decoder) # shape(None, 20, 100)

out = TimeDistributed(Dense(output_train.shape[2]))(decoder)
print(out) # shape(None, 20, 2)

model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

epc = 100
es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
history = model.fit('X_input_train', 'X_output_train', validation_split=0.2,
                     epochs=epc, verbose=1, callbacks=[es])
train_mae = history.history['mae']
valid_mae = history.history['val_mae']
 
model.save('model_forecasting_seq2seq.h5')

# 2. Seq2Seq LSTM Model with Luong Attention
'''
上述seq2seq的缺点
One of the limitations of simple Seq2Seq model is: only the last state of encoder RNN is used as input to decoder RNN.
If the sequence is very long, the encoder will tend to have much weaker memory about earlier time steps. 
Attention mechanism can solve this problem. An attention layer is going to assign proper weight to each hidden state 
output from encoder, and map them to output sequence.
'''
n_hidden = 100
input_train = Input(shape=(200, 2)) 
output_train = Input(shape=(20, 2))

## Encoder:输出stack_hidden_h用于计算attention
# For this Seq2Seq model, we also need stacked hidden states for alignment score calculation , last state_h and last state_c.
encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
                                                        return_sequences=True, return_state=True)(input_train)
print(encoder_stack_h) #  shape(None,  200, 100)
print(encoder_last_h) # shape(None,  100)
print(encoder_last_c) # shape(None, 100)

encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

## Decoder:输出stack_hidden_h用于计算attention
# repeat the last hidden state of encoder 20 times and use the as input to decoder
decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)  # (None, 100)--> (None, 20, 100)
# also need to get the stacked hidden state of decoder for alignment score calculation
decoder_stack_h = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
                       return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
print(decoder_stack_h) # (None, 20, 100)

## Attention Layer注意力层
'''
context = layers.Attention(use_scale=False, score_mode="dot")([decoder_stack_h,encoder_stack_h],\
    return_attention_scores=False,use_causal_mask=False)
'''
attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
attention = Activation('softmax')(attention)
print(attention)  #(None, 20, 200) # 注意力权重(batchsize, tgt_len, src_len)
context = dot([attention, encoder_stack_h], axes=[2, 1])
context = BatchNormalization(momentum=0.6)(context)
print(context)  # (None, 20, 100)

## concat context vector and stacked hidden states of decoder, and use it as input to the last dense layer
decoder_combined_context = concatenate([context, decoder_stack_h])
print(decoder_combined_context) # (None, 20, 200)

out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
print(out) # (None, 20, 2)

model = Model(inputs=input_train, outputs=out)
opt = Adam(lr=0.01, clipnorm=1)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


epc = 100
es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
history = model.fit('X_input_train', 'X_output_train', validation_split=0.2,
                     epochs=epc, verbose=1, callbacks=[es])
train_mae = history.history['mae']
valid_mae = history.history['val_mae']
 
model.save('model_forecasting_seq2seq.h5')