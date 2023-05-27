# https://blog.csdn.net/zengNLP/article/details/124375813?spm=1001.2014.3001.5502
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention, GRU, Dropout
from tensorflow.keras.models import Model
import numpy as np

'''
>>> inputs = tf.random.normal([32, 10, 8])
>>> gru = tf.keras.layers.GRU(4)
>>> output = gru(inputs)
>>> print(output.shape)
(32, 4)
>>> gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
>>> whole_sequence_output, final_state = gru(inputs)
>>> print(whole_sequence_output.shape)
(32, 10, 4)
>>> print(final_state.shape)
(32, 4)
'''

class Encoder(keras.Model):
    def __init__(self, hidden_units):
        super(Encoder, self).__init__()
        self.encoder_gru = GRU(hidden_units, return_sequences=True, return_state=True, name='encoder_gru')
        self.dropout = Dropout(rate=0.5)

    def call(self, inputs):
        encoder_outputs, state_h = self.encoder_gru(inputs)
        return encoder_outputs, state_h
    

class Decoder(keras.Model):
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

    opt = keras.optimizers.Adam()
    model.compile(loss=loss_fn, optimizer=opt)

    return model


model = seq2seq_attention((72,21), (24, 20), 50, 1)

def create_dataset(path):
    pass

def split_data(X1, X2, Y, test_size=0.2, shuffle=True):
    pass


def train(train_data_path, test_data_path):
    batch_size = 512
    epochs = 1000
    X1, X2, Y = create_dataset(train_data_path)
    train_data, eval_data, y_train, y_eval = split_data(X1, X2, Y, test_size=0.2, shuffle=True)
    
    
    #搭建模型
    model = seq2seq_attention((72,21), (24, 20), 50, 1)
    
    #训练模型
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath='model_path',
                                save_weights_only=True,
                                monitor='val_loss',
                                mode='min',
                                save_best_only=True)
    model.fit(x = train_data, 
                y = y_train, 
                batch_size = batch_size, 
                epochs = epochs, 
                callbacks=[callback, checkpoint_callback], 
                verbose = 2, 
                shuffle = True, 
                validation_data = (eval_data, y_eval))
    
	#模型测试
    X1, X2, Y = create_dataset(test_data_path)
    test_data = [X1, X2]
    y_test = Y
    scores = model.evaluate(test_data, y_test, verbose=0)
    print(model.metrics_names, scores)
	
