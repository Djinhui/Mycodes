import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 将数据传递到多输入、多输出模型
image_input = keras.Input(shape=(32,32,3), name='igm_imput')
ts_input = keras.Input(shape=(None, 10), name='ts_input')

x1 = layers.Conv2D(3,3)(image_input)
x1 = layers.GlobalMaxPooling2D()(x1)

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
