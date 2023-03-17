import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import * 

#打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))

    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)

MAX_LEN = 300
BATCH_SIZE = 32
(x_train, y_train),(x_test, y_test) = datasets.reuters.load_data()
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

MAX_WORDS = x_train.max() + 1
CAT_NUM = y_train.max() + 1

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()

ds_test = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE).cache()

# 1. 内置fit方法
tf.keras.backend.clear_session()
def create_model():
    model = models.Sequential()
    model.add(layers.Embedding(MAX_WORDS,7,input_length=MAX_LEN))
    model.add(layers.Conv1D(filters = 64,kernel_size = 5,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters = 32,kernel_size = 3,activation = "relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM,activation = "softmax"))
    return(model)

def compile_model(model:models.Sequential):
    model.compile(optimizer=optimizers.Nadam(),
                loss=losses.SparseCategoricalCrossentropy(),
                metrics=[metrics.SparseCategoricalAccuracy(),metrics.SparseTopKCategoricalAccuracy(5)]) 
    return(model)

model = create_model()
model.summary()
model = compile_model(model)
history = model.fit(ds_train,validation_data = ds_test,epochs = 10)


# 2. 内置train_on_batch方法
def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs+1):
        model.reset_metrics()

        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr / 2.0)

        for x, y in ds_train:
            train_result = model.train_on_batch()

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y,reset_metrics=False)

        if epoch%1 ==0:
            printbar()
            tf.print("epoch = ",epoch)
            print("train:",dict(zip(model.metrics_names,train_result)))
            print("valid:",dict(zip(model.metrics_names,valid_result)))
            print("")

model = create_model()
model = compile_model(model)
train_model(model,ds_train,ds_test,10)

# 3. 自定义训练循环
# 自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性。


optimizer = optimizers.Adam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_acc = metrics.SparseCategoricalAccuracy(name='train_Acc')

valid_loss = metrics.Mean(name='valid_loss')
valid_acc = metrics.SparseCategoricalAccuracy(name='valid_Acc')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    train_loss.update_state(loss)
    train_acc.update_state(labels, predictions)


@tf.function
def valid_step(model, features, labels):
    predictions = model(features, training=False)
    loss = loss_func(labels, predictions)
    valid_loss.update_state(loss)
    valid_acc.update_state(labels, predictions)


def train(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1,epochs+1):
        for featrues, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = 'Epoch={}, Loss:{}, Acc:{}, Valid loss:{}, Valid Acc:{}'
        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,(epoch, train_loss.result(), train_acc.result,\
                valid_loss.result, valid_acc.result)))

        # 每一轮后重置
        train_loss.reset_states()
        valid_loss.reset_states()
        train_acc.reset_states()
        valid_acc.reset_states()

model = create_model()
train_model(model, ds_train, ds_test, 20)
