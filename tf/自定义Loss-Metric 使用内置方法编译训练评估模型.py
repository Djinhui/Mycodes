import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. 简单示例
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are NumPy arrays)
x_train = x_train.reshape(60000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 编译 compile
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
# or 
def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


# 训练 fit
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# 评估与预测 evaluate and fit
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape) # (3, num_classes)


# 2.自定义损失
# 2.1 直接收y_true y_pred
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model = get_uncompiled_model()
model.compile(optimizer='adam', loss=custom_mse)

# 2.2 除y_true,y_pred, 还需要其他参数：Loss子类化
class CustomMSE(keras.losses.Loss):
    def __init__(self, reg_factor=0.1, name='cumtom_mse'): # 接受要在调用损失函数期间传递的参数
        super().__init__(name=name)
        self.reg_factor = reg_factor

    def call(self, y_true, y_pred): # 使用目标 (y_true) 和模型预测 (y_pred) 来计算模型的损失
        mse = tf.math.reduce_mean(tf.square(y_true, y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.reg_factor

model = get_uncompiled_model()
model.compile(optimizer='adam', loss=CustomMSE())

# 3. 自定义指标
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name='ctp', **kwargs): # 为您的指标创建状态变量
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positive = self.add_weight(name='ctp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None): #更新状态变量
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1,1))
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')
        if sample_weight:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)
        self.true_positive.assign_add(tf.reduce_sum(values))

    def result(self): # 使用状态变量来计算最终结果
        return self.true_positive

    def reset_state(self): # 重新初始化指标的状态
        # The state of the metric will be reset at the start of each epoch.
        self.true_positive.assign(0.0)

model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)

# 4. 处理不适合标准签名的损失和指标
# 4.1 从自定义层的调用方法内部调用 self.add_loss(loss_value), self.add_metric
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs  # Pass-through layer.

class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        # The `aggregation` argument defines
        # how to aggregate the per-batch values
        # over each epoch:
        # in this case we simply average them.
        self.add_metric(
            keras.backend.std(inputs), name="std_of_activation", aggregation="mean"
        )
        return inputs  # Pass-through layer.


inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)

# Insert activity regularization as a layer
x = ActivityRegularizationLayer()(x)

# Insert std logging as a layer.
x = MetricLoggingLayer()(x)

x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

# 4.2 在函数式API调用 model.add_loss(loss_tensor) 或 model.add_metric(metric_tensor, name, aggregation)。
inputs = keras.Input(shape=(784,), name="digits")
x1 = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x2 = layers.Dense(64, activation="relu", name="dense_2")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)

model.add_loss(tf.reduce_sum(x1) * 0.1)
model.add_metric(keras.backend.std(x1), name="std_of_activation", aggregation="mean")
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)
model.fit(x_train, y_train, batch_size=64, epochs=1)

# 4.3 当通过add_loss()传递损失时，可以在没有损失函数的情况下调用 compile()，因为模型已经有损失要最小化
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.acc_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()
        acc = self.acc_fn(targets, logits, sample_weights)
        self.add_metric(acc, name='acc')

        # Return the inference-time prediction tensor (for `.predict()`)
        return tf.nn.softmax(logits)

import numpy as np

inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")  # No loss argument!

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)
    
# 5. 自动分离验证集 validation_split
model = get_compiled_model()
model.fit(x_train, y_train, batch_size=64, validation_split=0.2, epochs=1)

# 6. 通过 tf.data 数据集进行训练和评估 tf.data.Dataset.from_tensor_slices((x,y,sample_weights)).shuffle().batch()
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64) # no shuffle

model.fit(train_dataset, epochs=3, steps_per_epoch=100)
result = model.evaluate(test_dataset)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)
model.fit(train_dataset, epochs=1, validation_data=val_dataset, validation_steps=10)

# 7. 使用 keras.utils.Sequence 对象作为输入 __getitem__ & __len__
from skimage.io import imread
from skimage.transform import resize

# Here, `filenames` is list of path to the images
# and `labels` are the associated labels.

class CIFAR10Sequence(keras.utils.Sequence):
    def __init__(self, filenames, labels,batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx*self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(imread(filename), (200, 200))for filename in batch_x]), \
            np.array(batch_y)

filenames = ['']
labels = ['']
batch_size = 64
sequence = CIFAR10Sequence(filenames, labels, batch_size)
model.fit(sequence, epochs=10)


# 8. 样本权重和类权重
class_weight = {
    0: 1.0,
    1: 1.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    # Set weight "2" for class "5",
    # making this class 2x more important
    5: 2.0,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
}
sample_weight = np.ones(shape=(len(y_train),))
sample_weight[y_train == 5] = 2.0

model = get_compiled_model()
model.fit(x_train, y_train, class_weight=class_weight, batch_size=64, epochs=1)
model.fit(x_train, y_train, sample_weight=sample_weight, batch_size=64, epochs=1)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train, sample_weight))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model = get_compiled_model()
model.fit(train_dataset, epochs=1)

# 9. 将数据传递到多输入、多输出模型
image_input = keras.Input(shape=(32,32,3), name='igm_imput')
ts_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x)

x2 = layers.Conv1D(3,3)(ts_input)
x2 = layers.GlobalMaxPooling1D()(x2)

x = layers.concatenate([x1,x2])

score_output = layers.Dense(1, name='score_output')(x)
class_output = layers.Dense(5, name='class_output')(x)

model = keras.Model(inputs=[image_input, ts_input], outputs=[score_output, class_output])
keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# 9.1 在编译时，通过将损失函数作为列表传递，我们可以为不同的输出指定不同的损失
# 如果我们仅将单个损失函数传递给模型，则相同的损失函数将应用于每个输出（此处不合适）
# 对于指标同样如此
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[keras.losses.MeanSquaredError(), keras.losses.CategoricalCrossentropy()],
    metrics=[
        [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        [keras.metrics.CategoricalAccuracy()],
    ],
)

# 9.2 由于我们已为输出层命名，我们还可以通过字典指定每个输出的损失和指标
# 并且使用 loss_weights 参数为特定于输出的不同损失赋予不同的权重
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={
        "score_output": keras.losses.MeanSquaredError(),
        "class_output": keras.losses.CategoricalCrossentropy(),
    },
    metrics={
        "score_output": [
            keras.metrics.MeanAbsolutePercentageError(),
            keras.metrics.MeanAbsoluteError(),
        ],
        "class_output": [keras.metrics.CategoricalAccuracy()],
    },
    loss_weights={"score_output": 2.0, "class_output": 1.0},
)

# 9.3 如果这些输出用于预测而不是用于训练，也可以选择不计算某些输出的损失
# List loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=[None, keras.losses.CategoricalCrossentropy()],
)

# Or dict loss version
model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss={"class_output": keras.losses.CategoricalCrossentropy()},
)

# 9.4 将数据传递给 fit() 中的多输入或多输出模型的工作方式与在编译中指定损失函数的方式类似：
# 您可以传递 NumPy 数组的列表（1:1 映射到接收损失函数的输出），或者通过字典将输出名称映射到 NumPy 数组
# Generate dummy NumPy data
img_data = np.random.random_sample(size=(100, 32, 32, 3))
ts_data = np.random.random_sample(size=(100, 20, 10))
score_targets = np.random.random_sample(size=(100, 1))
class_targets = np.random.random_sample(size=(100, 5))

# Fit on lists
model.fit([img_data, ts_data], [score_targets, class_targets], batch_size=32, epochs=1)

# Alternatively, fit on dicts
model.fit(
    {"img_input": img_data, "ts_input": ts_data},
    {"score_output": score_targets, "class_output": class_targets},
    batch_size=32,
    epochs=1,
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"img_input": img_data, "ts_input": ts_data},
        {"score_output": score_targets, "class_output": class_targets},
    )
)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
model.fit(train_dataset, epochs=1)

# 10. 使用回调 keras.callbacks.*
# 自定义回调
class HistoryCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []
    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get('loss'))

# 11. 学习率调度keras.optimizers.schedules.*
lr_init = 0.1
lr_schedule = keras.optimizers.schedules.ExponentialDecay(lr_init, decay_steps=1000)
optimzizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)


