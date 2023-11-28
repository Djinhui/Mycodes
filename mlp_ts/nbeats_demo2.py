import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from nbeats_keras.model import NBeatsNet

# path = 'AirPassengers.csv'
# data = pd.read_csv(path)
# data['AirPassengers'] = StandardScaler().fit_transform(data['AirPassengers'].values.reshape(-1,1))
# plt.plot(data['AirPassengers'],label='AirPassengers')

# path = 'PJM_Load_hourly.csv'
# data = pd.read_csv(path)
# data['PJM_Load_MW'] = StandardScaler().fit_transform(data['PJM_Load_MW'].values.reshape(-1,1))
# plt.plot(data['PJM_Load_MW'],label='PJM_Load_MW')


path = 'guanglin.csv'
data = pd.read_csv(path)
data['y'] = StandardScaler().fit_transform(data['y'].values.reshape(-1,1))
plt.plot(data['y'],label='y')

plt.legend()
plt.show()

def data_generation(data, length):
    shape = data.shape[0]
    for start, stop in zip(range(0, shape-length), range(length, shape)):
        yield data[start:stop].reshape(-1,1), data[stop]


backcast_length =  96 * 1


X, Y = [], []
for x, y in data_generation(data['y'].values, length=backcast_length):
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y).reshape((len(Y), 1))

print(X.shape, Y.shape) # (num_samples, lookback, 1), (num_samples, 1)

# print(X[0])
# print(X[1])
# print(Y[0])

X_train, X_val = X[:int(X.shape[0]*0.8)], X[int(X.shape[0]*0.8):]
Y_train, Y_val = Y[:int(Y.shape[0]*0.8)], Y[int(Y.shape[0]*0.8):]

nbeats = NBeatsNet(input_dim=1, output_dim=1, backcast_length=backcast_length, forecast_length=1,
                   stack_types=(NBeatsNet.TREND_BLOCK, NBeatsNet.GENERIC_BLOCK),
                   nb_blocks_per_stack=3, thetas_dim=(4,4), share_weights_in_stack=True, hidden_layer_units=64)

nbeats.compile(loss='mae', optimizer=Adam(learning_rate=1e-3))
his = nbeats.fit(X_train, Y_train, epochs=50, batch_size=64, validation_split=0.2)

plt.figure(figsize=(12, 7))
plt.plot(his.history['loss'], label='train loss')
plt.plot(his.history['val_loss'], label='val loss')
plt.legend()
plt.show()

nbeats.enable_intermediate_outputs()
res = nbeats.predict(X_val)
plt.plot(res.reshape(len(Y_val)), label='predict')
plt.plot(Y_val, label='real')
plt.legend()
plt.show()


res = nbeats.predict(X_val)
plt.plot(res.reshape(len(Y_val))[:96*3], label='predict')
plt.plot(Y_val[:96*3], label='real')
plt.legend()
plt.show()


# generic, interpretable, sums = [], [], []
# for i in range(len(X_val)):
#     # nbeats.predict(X_val[i].reshape((1, backcast_length, 1)))
#     tmp = nbeats.get_generic_and_interpretable_outputs()
#     generic.append(tmp[0])
#     interpretable.append(tmp[1])
#     sums.append(tmp[0] + tmp[1])

# plt.plot(generic, label='generic')
# plt.plot(interpretable, label='interpretable')
# plt.plot(sums, label='generic+interpretable')
# plt.plot(res.reshape(len(Y_val)),label='predict')
# plt.plot(Y_val,label='rea1')
# plt.legend()
# plt.show()