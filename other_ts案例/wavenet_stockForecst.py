# https://www.kaggle.com/code/bhavinmoriya/conv1d-wavenet-forecast-stock-price/notebook

# see at https://github.com/philipperemy for TCN and Attention

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 20, 15

ticker = 'bidi4.sa'
start = dt.date(2015,1,1)
df = web.get_data_yahoo(ticker, start=start)
df = df.Close
plt.plot(df)

df = df.values
df = df.reshape(len(df), 1)

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
# total 904 data

def rnn_process(scaled_data, window=60, test_size=0.2):
    test_len = int(len(scaled_data) * test_size)
    train = scaled_data[:-test_len]
    test = scaled_data[-test_len-window:]

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(window, len(train)):
        X_train.append(train[i-window:i, 0])
        y_train.append(train[i,0])

    for i in range(window, len(test)):
        X_test.append(test[i-window:i, 0])
        y_test.append(test[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    # (664, 60, 1), (664,), (180, 60, 1), (180,)
    print(f'Shapes of the X_train, y_train, X_test, y_test are {X_train.shape, y_train.shape, X_test.shape, y_test.shape}')
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = rnn_process(scaled_data)

model = keras.models.Sequential()
model.add(keras.layers.InputLayer(input_shape=[60, 1]))
# https://github.com/ageron/handson-ml2/blob/master/15_processing_sequences_using_rnns_and_cnns.ipynb
# 简化版的wavenet
for rate in (1,2,4,8)*2:  
    model.add(keras.layers.Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', dilation_rate=rate))

model.add(keras.layers.Conv1D(filters=10, kernel_size=1))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mse', optimizer='adam')
model.summary()

checkpts = ModelCheckpoint('./conv1d.h5', verbose=1, save_best_only=True)
es = EarlyStopping(patience=5)
callbacks = [checkpts, es]

history = model.fit(X_train, y_train, epochs=20, verbose=1, batch_size=32, validation_split=.2, callbacks=callbacks)

plt.plot(history.history['loss'], label='Original loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()


model.evaluate(X_test, y_test, batch_size=32)


tomorrow_pred = scaler.inverse_transform(model.predict(scaled_data[-60:].reshape(1,60,1)))
print(f'Prediction for tomorrow is {tomorrow_pred[0,0]}')

# Last day value 
scaler.inverse_transform(scaled_data[-1].reshape(-1,1)), df[-1] 


# 迭代预测10天
def forecast(model, data, future=10, window=60):
    # function works for the window size 60 only. data must be last 60 days values
    predictions = []
    for i in range(future):
        predictions.append(model.predict(data.reshape(1, window, 1)))
        data = np.concatenate((data, predictions[-1]), axis=0)[-window:]

    return np.array(predictions).reshape(-1,1)


future = forecast(model, scaled_data[-60:])
future = scaler.inverse_transform(future)

tomorrow = dt.date.today() + dt.timedelta(days=1)
period = pd.date_range(tomorrow, periods=10, freq='B')
forecast = pd.DataFrame(future, columns=['Forecast'])
forecast = forecast.set_index(period)
forecast

plt.plot(web.get_data_yahoo(ticker, start = start).Close[-10:])
plt.plot(forecast)