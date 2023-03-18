import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from sklearn.preprocessing import MinMaxScaler

maotai = pd.read_csv('maotai.csv')

train_set = maotai.iloc[0:14000, 1:2].values
test_set = maotai.iloc[14000:15000, 1:2].values

scaler = MinMaxScaler(feature_range=(0, 1))
train_set_scaled = scaler.fit_transform(train_set)
test_set_scaled = scaler.transform(test_set)

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(60, 14000):
    x_train.append(train_set_scaled[i-60:i, 0])
    y_train.append(train_set_scaled[i, 0])

for i in range(60, len(test_set)):
    x_test.append(test_set_scaled[i-60:i, 0])
    y_test.append(test_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], 60,1))
x_test = np.reshape(x_test, (x_test.shape[0], 60,1))

model = tf.keras.models.Sequential([
    SimpleRNN(80, return_sequences=True),
    SimpleRNN(80),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
# checkpoint_save_path = './checkpoint/SimpleRNN.ckpt'
# if os.path.exists(checkpoint_save_path + '.index'):
#     model.load_weights(checkpoint_save_path)

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#     save_weights_only=True,
#     save_best_only=True, monitor='loss')

history = model.fit(x_train, y_train, epochs=50, batch_size=32,)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
real = scaler.inverse_transform(x_test[60:])



