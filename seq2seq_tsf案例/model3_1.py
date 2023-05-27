# https://blog.csdn.net/Cyril_KI/article/details/126563659
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Encoder(keras.Model):
    def __init__(self, args):
        super().__init__()
        self.lstm = layers.LSTM(units=args.hidden_size, input_shape=(args.seq_len, args.input_size),
                                activation='tanh', return_sequences=True, return_state=True)
    def call(self, input_seq):
        output, h, c = self.lstm(input_seq)
        return h, c
    

class Decoder(keras.Model):
    def __init__(self, args):
        self.lstm = layers.LSTM(units=args.hidden_size, input_shape=(args.seq_len, args.input_size),
                                activation='tanh', return_sequences=True, return_state=True)
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(args.output_size)

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


class Seq2Seq(keras.Model):
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
