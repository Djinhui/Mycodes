import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Layer类
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype='float32'),trainabel=True)

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units, ), dtype='float32'),trainable=True)

    def call(self, x):
        return tf.matmul(x, self.w) + self.b

# 1.1更加快捷的方式为层添加权重：add_weight() 方法
class Linear(layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(units, ), initializer='zeros', trainable=True)

    def call(self, x):
        return tf.matmul(x, self.w) + self.b

# 调用层
x = tf.ones((2, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)

assert linear_layer.weights == [linear_layer.w, linear_layer.b]

# 1.2. 不用训练的权重
class ComputSum(layers.Layer):
    def __init__(self, input_dim):
        super(ComputSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim, )), trainable=False)

    def call(self, x):
        self.total.assign_add(tf.reduce_sum(x, axis=0))
        return self.total

x = tf.ones((2, 2))
my_sum = ComputSum(2)
y = my_sum(x)
print(y.numpy()) # [2,2]
y = my_sum(x)
print(y.numpy()) # [4,4]

print("weights:", len(my_sum.weights)) # 1
print("non-trainable weights:", len(my_sum.non_trainable_weights)) # 1
print("trainable_weights:", my_sum.trainable_weights) # []

# 1.3. 将权重创建推迟到得知输入的形状之后 build()
class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

# At instantiation, we don't know on what inputs this is going to get called
linear_layer = Linear(32)

# The layer's weights are created dynamically the first time the layer is called
y = linear_layer(x)

# 1.4. 层可递归组合
class MLPBlock(keras.layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x = tf.nn.relu(x)
        return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print("weights:", len(mlp.weights)) # 6
print("trainable weights:", len(mlp.trainable_weights)) # 6

# 1.5. 层的self.add_loss()方法  通过 layer.losses 取回

# A layer that creates an activity regularization loss
class ActivityRegularizationLayer(keras.layers.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs

class OuterLayer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs):
        return self.activity_reg(inputs)


layer = OuterLayer()
assert len(layer.losses) == 0  # No losses yet since the layer has never been called

_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # We created one loss value

# `layer.losses` gets reset at the start of each __call__
_ = layer(tf.zeros(1, 1))
assert len(layer.losses) == 1  # This is the loss created during the call above

# 1.5.1 loss 属性还包含为任何内部层的权重创建的正则化损失，在编写训练循环时应考虑这些损失
class OuterLayerWithKernelReg(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        return self.dense(inputs)

layer = OuterLayerWithKernelReg()
_ = layer(tf.zeros((1, 1)))

# This is `1e-3 * sum(layer.dense.kernel ** 2)`,
# created by the `kernel_regularizer` above.
print(layer.losses)


# for x_batch_train, y_batch_train in train_dataset:
#   with tf.GradientTape() as tape:
#     logits = layer(x_batch_train)  # Logits for this minibatch
#     loss_value = loss_fn(y_batch_train, logits)
#     # Add extra losses created during this forward pass:
#     loss_value += sum(model.losses) # 在编写训练循环时应考虑这些损失

#   grads = tape.gradient(loss_value, model.trainable_weights)
#   optimizer.apply_gradients(zip(grads, model.trainable_weights))

# 1.5.2 自定义损失如果使用fit(), 它们会自动求和并添加到主损失中
import numpy as np

inputs = keras.Input(shape=(3,))
outputs = ActivityRegularizationLayer()(inputs)
model = keras.Model(inputs, outputs)

# 1.5.2.1 If there is a loss passed in `compile`, the regularization losses get added to it
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# 1.5.2.2 It's also possible not to pass any loss in `compile`,
# since the model already has a loss to minimize, via the `add_loss`
# call during the forward pass!
model.compile(optimizer="adam")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# 1.6. 层的add_metric()方法， 用于在训练过程中跟踪数量的移动平均值

# Demo:它将预测和目标作为输入，计算通过 add_loss() 跟踪的损失，并计算通过 add_metric() 跟踪的准确率标量
class LogisticEndpoint(keras.layers.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        # Log accuracy as a metric and add it
        # to the layer using `self.add_metric()`.
        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        # Return the inference-time prediction tensor (for `.predict()`).
        return tf.nn.softmax(logits)

layer = LogisticEndpoint()

targets = tf.ones((2, 2))
logits = tf.ones((2, 2))
y = layer(targets, logits)

print("layer.metrics:", layer.metrics)
print("current accuracy value:", float(layer.metrics[0].result()))

# or use fit()
inputs = keras.Input(shape=(3,), name="inputs")
targets = keras.Input(shape=(10,), name="targets")
logits = keras.layers.Dense(10)(inputs)
predictions = LogisticEndpoint(name="predictions")(logits, targets)

model = keras.Model(inputs=[inputs, targets], outputs=predictions)
model.compile(optimizer="adam")

data = {
    "inputs": np.random.random((3, 3)),
    "targets": np.random.random((3, 10)),
}
model.fit(data)


# 1.7. 序列化自定义层， 实现get_config()方法

# 基础 Layer 类的 __init__() 方法会接受一些关键字参数，尤其是 name 和 dtype。
# 最好将这些参数传递给 __init__() 中的父类super(Linear, self).__init__(**kwargs)，
# 并将其包含在层配置中super(Linear, self).get_config()
class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )
        # super(Linear,self).build(input_shape) # 相当于设置self.built = True
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config

    def from_config(cls, config):
        return cls(**config)

layer = Linear(64)
config = layer.get_config()
print(config)
new_layer = Linear.from_config(config)

# linear = Linear(units = 8)
# print(linear.built)
# #指定input_shape，显式调用build方法，第0维代表样本数量，用None填充
# linear.build(input_shape = (None,16)) 
# print(linear.built)
# print(linear.compute_output_shape(input_shape = (None,16)))

# 1.8. call()方法的特色参数 training 和mask

# 某些层，尤其是 BatchNormalization 层和 Dropout 层，在训练和推断期间具有不同的行为。
# 对于此类层，标准做法是在 call() 方法中公开 training（布尔）参数。
class CustomDropout(keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


# 2. 自定义Model
'''
Model 类具有与 Layer 相同的 API,但有如下区别：

它会公开内置训练、评估和预测循环(model.fit()、model.evaluate()、model.predict())。
它会通过 model.layers 属性公开其内部层的列表。
它会公开保存和序列化 API(save()、save_weights()…)

实际上Layer 类对应于我们在文献中所称的“层”(如“卷积层”或“循环层”或“块”(如“ResNet 块”或“Inception 块”)
同时Model 类对应于文献中所称的“模型”（如“深度学习模型”）或“网络”（如“深度神经网络”）。

通常，您会使用 Layer 类来定义内部计算块，并使用 Model 类来定义外部模型，即您将训练的对象
是否需要在它上面调用 fit()？我是否需要在它上面调用 save()？如果是，则使用 Model。
如果不是（要么因为您的类只是更大系统中的一个块，要么因为您正在自己编写训练和保存代码），则使用 Layer

'''

# class ResNet(tf.keras.Model):

#     def __init__(self, num_classes=1000):
#         super(ResNet, self).__init__()
#         self.block_1 = ResNetBlock()
#         self.block_2 = ResNetBlock()
#         self.global_pool = layers.GlobalAveragePooling2D()
#         self.classifier = Dense(num_classes)

#     def call(self, inputs):
#         x = self.block_1(inputs)
#         x = self.block_2(x)
#         x = self.global_pool(x)
#         return self.classifier(x)


# resnet = ResNet()
# dataset = ...
# resnet.fit(dataset, epochs=10)
# resnet.save(filepath)


# 将这些内容全部汇总到一个端到端示例：我们将实现一个变分自动编码器 (VAE)，并用 MNIST 数字对其进行训练
# see [vae.py]