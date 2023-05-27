import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib


def code_mean(df, cat_feature, target_col):
    return df.groupby(cat_feature)[target_col].mean().to_dict()

def lags_windows(df, target_coding=False, lags=[1,2,3,4], wins=[5], train_en_dt='2023-03-01'):
    lag_cols = ['lag_{}'.format(lag) for lag in lags]
    for lag, lag_col in zip(lags, lag_cols):
        df[lag_col] = df['y'].shift(lag)
        
    for win in wins:
        for lag, lag_col in zip(lags, lag_cols):
            df['rmean_{}_{}'.format(lag, win)] = df[lag_col].rolling(win).mean()
            df['rmedian_{}_{}'.format(lag, win)] = df[lag_col].rolling(win).median()
            df['rmax_{}_{}'.format(lag, win)] = df[lag_col].rolling(win).max()
            df['rmin_{}_{}'.format(lag, win)] = df[lag_col].rolling(win).min()
            df['rstd_{}_{}'.format(lag, win)] = df[lag_col].rolling(win).std()
            
    if target_coding:
        # calculate on train set only
        temp_train_val = df[df['ds'] < train_en_dt].dropna()
        length = int(temp_train_val.shape[0] * 0.9)
        temp_train = temp_train_val.iloc[:length,:]
        
        df['weekday_mean'] = list(map(code_mean(temp_train, 'weekday', 'y').get, df.weekday))
        
    return df

def draw(data1, data2):
    ax = plt.subplot(111)
    ax.plot(list(range(len(data1))), data1, 'b-', list(range(len(data1), len(data1)+len(data2))), data2, 'r-')
    plt.show()


def clac_acc(y_true, y_hat):
    e = np.abs(y_hat - y_true) / y_true
    e = np.where(e > 1.0, 0.9, e)
    a = 1 - np.sqrt(np.dot(e,e) / len(e))
    print('acc: ', a)
    
def elaluate_loss(y, pred):
    pass
    # print('r2 score:{:<8.5f}'.format(r2_score(y, pred)))
    # print("mae: {:<8.5f}".format(mean_absolute_error(y, pred)))
    
def plot(y, pred, st=0, ed=10):
    plt.figure(figsize=(20,6))
    plt.plot(range(len(y[st:ed])),y[st:ed],color='g',label='true')
    plt.plot(range(len(pred[st:ed])),pred[st:ed],color='r', label='pred')
    plt.legend()
    plt.show()
    

def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_pred - y_true) / y_true))

def regression_metrics(y_true, y_pred):
    pass
    # print('均方误差[MSE]:', mean_squared_error(y_true, y_pred))
    # print('均方根误差[RMSE]:', np.sqrt(mean_squared_error(y_true, y_pred)))
    # print('平均绝对误差[MAE]:', mean_absolute_error(y_true, y_pred))
    # print('绝对误差中位数[MedianAE]:', median_absolute_error(y_true, y_pred))
    # print('平均绝对百分比误差[MAPE]:',mean_absolute_percentage_error(y_true, y_pred))
    # print('绝对百分比误差中位数[MedianAPE]:', median_absolute_percentage_error(y_true, y_pred))



def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=32, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
        
    i = min_index + lookback
    while True:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)
            
        samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
        if delay == 1:
            targets = np.zeros((len(rows), delay))
        else:
            targets = np.zeros((len(rows), delay, 1))
            
        for j, row in enumerate(rows):
            indices = range(rows[j]-lookback, rows[j], step)
            samples[j] = data[indices]
            if delay == 1:
                targets[j] = data[rows[j]:rows[j]+delay]
            else:
                targets[j] = np.array(data[rows[j]:rows[j]+delay]).reshape(-1,1)
                
        yield samples, targets


# kernel_regularizer=regularizers.l2(0.01),recurrent_dropout=0.5
#  dropout=0.2, recurrent_dropout=0.2,kernel_regularizer=regularizers.l2(0.1)
def lstm_model2():
    model = Sequential()
    model.add(LSTM(32,return_sequences=True, input_shape=(lookback, 1)))
#     model.add(LSTM(32, input_shape=(lookback, 1)))
    model.add(LSTM(32))
    model.add(RepeatVector(delay))
    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    
    opt = Adam(lr=0.001) # clipnorm=1.0
#     opt = SGD(lr=0.1) # momentum=0.9,nesterov=True
    model.compile(optimizer=opt, loss='mae')
    print(model.summary())
    return model

def train_model2():
    model = lstm_model2()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='min', factor=0.5, min_delta=0.001)
    es = EarlyStopping(monitor='val_loss', mode='min',verbose=1, patience=10)
    
    date2str = datetime.now().strftime('%Y%m%d %H:%M:%S')
    date2str = date2str.replace(' ','').replace(':','')
    
    ckpt = ModelCheckpoint('batchsize_{}.h5'.format(batch_size),monitor='val_loss',
                          verbose=0, save_best_only=True,save_weights_only=False, mode='min', period=1)
    history = model.fit_generator(train_gen, steps_per_epoch=train_steps, epochs=100, validation_data=val_gen,
                                 validation_steps=val_steps, callbacks=[ckpt,es])
    
    return model, history



lookback = 96
step = 1
delay = 24
batch_size = 64
r1 = 96 * int(0.91 * 631)
r2 = 96 * int(0.953 * 631)


float_data = np.arange(100000)
min_max_scaler_y = 'MinMaxScaler().fit_t(float_data)'


train_gen = generator(data=float_data, lookback=lookback, delay=delay, min_index=0,max_index=r1,
                     shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(data=float_data, lookback=lookback, delay=delay, min_index=r1,max_index=r2,
                    shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(data=float_data, lookback=lookback, delay=delay, min_index=r2,max_index=None,
                     shuffle=False, step=step, batch_size=batch_size)

train_steps = r1 // batch_size - 1
# this is how many steps to draw from val_gen in order to see the whole validation set
val_steps = (r2 - r1 - lookback) // batch_size
test_steps = (len(float_data) - r2 - lookback) // batch_size


K.clear_session()
model, history = train_model2()

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs[:], loss[:], 'r', label='Training loss')
plt.plot(epochs[:], val_loss[:], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save('batchsize_{}_all.h5'.format(batch_size))
joblib.dump(min_max_scaler_y, 'max_min_scaler_batchsize_{}.model'.format(batch_size))
samples, targets = next(val_gen)
print(samples.shape, targets.shape)

np.save('samples_batchsize_{}.npy'.format(batch_size), samples)
cls = {'samples':samples, 'max_min_scaler': min_max_scaler_y, 'n_features': 1,'look_back':96, 'delay':24,}
joblib.dump(cls,'basic_params')

# del model
batch_size = 64

# del model1, model_all, min_max_scaler1, samples
model1 = load_model('batchsize_{}.h5'.format(batch_size))
model_all = load_model('batchsize_{}_all.h5'.format(batch_size))
min_max_scaler1 = joblib.load('max_min_scaler_batchsize_{}.model'.format(batch_size))
samples = np.load('samples_batchsize_{}.npy'.format(batch_size))


# 训练集

pred_all_tr = np.array([])
y_true_tr = np.array([])

for i in range(0, 55104, 96): 
    k = i
    
    def predict_seq2seq_multi_steps(model, samples, n_features, lookback, delay, steps):
        samples[0] = float_data[k:lookback+k].reshape(lookback, n_features) # (64, 96, 1)
        pred = np.empty(shape=[0,steps])
        while steps:
            t = model.predict(samples) # (64, 24, 1)
            pred = np.append(pred, np.squeeze(t)[0]) # get the first pred result,  lenth of pred iteratly -> [24,,48,72,96]
            tmp = np.append(np.squeeze(samples[0]), np.squeeze(t)[0])
            samples[0] = tmp[delay:].reshape(lookback, n_features)
            steps -= 1
            
#         samples=float_data[k:lookback+k].reshape(1,lookback,n_features) # (1, 96, 1)
#         pred=np.empty(shape=[0,steps])
#         while steps:
#             t=model.predict(samples)  # (1,,24,1)
#             pred=np.append(pred,np.squeeze(t))
#             tmp=np.append(np.squeeze(samples),np.squeeze(t))
#             samples=tmp[delay:].reshape(1,lookback,n_features)
#             steps-=1


        return pred
    
    pred = predict_seq2seq_multi_steps(model1, samples, 1, lookback, delay, steps=4*1)
    st = 0
    ed = len(pred)
    
    y_true_tr = np.append(y_true_tr, np.squeeze(min_max_scaler1.inverse_transform(float_data[lookback+k:][st:ed])))
    pred_all_tr = np.append(pred_all_tr, np.squeeze(min_max_scaler1.inverse_transform(pred[:][st:ed].reshape(-1,1))))
    plt.figure(figsize=(20,6))
    plt.plot(min_max_scaler1.inverse_transform(float_data[lookback+k:][st:ed]), 'g', label='true')
    plt.plot(min_max_scaler1.inverse_transform(pred[:][st:ed].reshape(-1, 1)), 'b', label='pred')
    plt.legend()
    plt.show()

print('=======train result======')
clac_acc(y_true_tr, pred_all_tr)
elaluate_loss(y_true_tr, pred_all_tr)
regression_metrics(y_true_tr, pred_all_tr)
plot(y_true_tr, pred_all_tr,0,ed=len(pred_all_tr))

# 验证集

pred_all_val = np.array([])
y_true_val = np.array([])

for i in range(0, 2592, 96): 
    k = i
    def predict_seq2seq_multi_steps(model, samples, n_features, lookback, delay, steps):
        samples[0] = float_data[r1+k:r1+k+lookback].reshape(lookback, n_features)
        pred = np.empty(shape=[0,steps])
        while steps:
            t = model.predict(samples)
            pred = np.append(pred, np.squeeze(t)[0])
            tmp = np.append(np.squeeze(samples[0]), np.squeeze(t)[0])
            samples[0] = tmp[delay:].reshape(lookback, n_features)
            steps -= 1
        return pred
    
    pred = predict_seq2seq_multi_steps(model1, samples, 1, lookback, delay, steps=4*1)
    st = 0
    ed = len(pred)
    
    y_true_val = np.append(y_true_val, np.squeeze(min_max_scaler1.inverse_transform(float_data[r1+lookback+k:][st:ed])))
    pred_all_val = np.append(pred_all_val, np.squeeze(min_max_scaler1.inverse_transform(pred[:][st:ed].reshape(-1,1))))
    plt.figure(figsize=(20,6))
    plt.plot(min_max_scaler1.inverse_transform(float_data[r1+lookback+k:][st:ed]), 'g', label='true')
    plt.plot(min_max_scaler1.inverse_transform(pred[:][st:ed].reshape(-1, 1)), 'b', label='pred')
    plt.legend()
    plt.show()