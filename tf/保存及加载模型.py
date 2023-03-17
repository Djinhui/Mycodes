import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import regularizers, initializers
'''
保存
model = ...  # Get model (Sequential, Functional Model, or Model subclass)
model.save('path/to/location')
加载
model = keras.models.load_model('path/to/location')
model.predict(test)
'''


# 1. 全模型保存及加载
# model.save() 或 tf.keras.models.save_model()
# tf.keras.models.load_model()

# 1.1 保存格式：TensorFlow SavedModel 格式和较早的 Keras H5 格式
'''
SavedModel 是更全面的保存格式，它可以保存模型架构、权重和调用函数的跟踪 Tensorflow 子计算图,
这使 Keras 能够恢复内置层和自定义对象;

与 SavedModel 格式相比,H5 文件不包括以下两方面内容：

1. 通过 model.add_loss() 和 model.add_metric() 添加的外部损失和指标不会被保存（这与 SavedModel 不同）。
如果您的模型有此类损失和指标且您想要恢复训练，则您需要在加载模型后自行重新添加这些损失。
请注意，这不适用于通过 self.add_loss() 和 self.add_metric() 在层内创建的损失/指标。
只要该层被加载，这些损失和指标就会被保留，因为它们是该层 call 方法的一部分。
2. 已保存的文件中不包含自定义对象(如自定义层)的计算图。
在加载时,Keras 需要访问这些对象的 Python 类/函数以重建模型。
'''
# 1.1.1 推荐使用 SavedModel 格式。它是使用 model.save() 时的默认格式
def get_model():
    inputs = keras.Input(shape=(32,))
    outputs = keras.layers.Dense(1)(inputs)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

model = get_model()
test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))
model.fit(test_input, test_target)

# Calling `save('my_model')` creates a SavedModel folder `my_model`.
# 创建一个名为 my_model 的文件夹
model.save("my_model")
# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")
reconstructed_model.fit(test_input, test_target)

# 保存自定义的模型
class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.hidden_units = hidden_units
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

    def get_config(self):
        return {"hidden_units": self.hidden_units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save("my_model")

# Option 1: Load with the custom_object argument.
# 使用配置和 CustomModel 类加载
loaded_1 = keras.models.load_model(
    "my_model", custom_objects={"CustomModel": CustomModel}
)

# Option 2: Load without the CustomModel class.
# Delete the custom-defined model class to ensure that the loader does not have access to it.
# 通过动态创建类似于原始模型的模型类来加载
del CustomModel
loaded_2 = keras.models.load_model("my_model")
np.testing.assert_allclose(loaded_1(input_arr), outputs)
np.testing.assert_allclose(loaded_2(input_arr), outputs)

print("Original model:", model)
print("Model Loaded with custom objects:", loaded_1)
print("Model loaded without the custom object class:", loaded_2)


# 1.1.2切换到 H5 格式：将 save_format='h5' 传递给 save()；将以 .h5 或 .keras 结尾的文件名传递给 save()
# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
model.save("my_h5_model.h5")
# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_h5_model.h5")

# 2. 保存架构
'''
模型的配置（或架构）指定模型包含的层，以及这些层的连接方式。
如果您有模型的配置，则可以使用权重的新初始化状态创建模型，而无需编译信息。
*请注意，这仅适用于使用函数式或序列式 API 定义的模型，不适用于子类化模型
get_config() 和 from_config()
tf.keras.models.model_to_json() 和 tf.keras.models.model_from_json()

'''

# 3. 只需使用模型进行推断 
'''
在内存中将权重从一层转移到另一层

您只需使用模型进行推断：在这种情况下，您无需重新开始训练，因此不需要编译信息或优化器状态。
您正在进行迁移学习：在这种情况下，您需要重用先验模型的状态来训练新模型，因此不需要先验模型的编译信息。
tf.keras.layers.Layer.get_weights()：返回 Numpy 数组列表。
tf.keras.layers.Layer.set_weights()：将模型权重设置为 weights 参数中的值

'''
# Create a simple functional model
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

# Define a subclassed model with the same architecture
class SubclassedModel(keras.Model):
    def __init__(self, output_dim, name=None):
        super(SubclassedModel, self).__init__(name=name)
        self.output_dim = output_dim
        self.dense_1 = keras.layers.Dense(64, activation="relu", name="dense_1")
        self.dense_2 = keras.layers.Dense(64, activation="relu", name="dense_2")
        self.dense_3 = keras.layers.Dense(output_dim, name="predictions")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x

    def get_config(self):
        return {"output_dim": self.output_dim, "name": self.name}


subclassed_model = SubclassedModel(10)
# Call the subclassed model once to create the weights.
subclassed_model(tf.ones((1, 784)))

# Copy weights from functional_model to subclassed_model.
subclassed_model.set_weights(functional_model.get_weights())

assert len(functional_model.weights) == len(subclassed_model.weights)
for a, b in zip(functional_model.weights, subclassed_model.weights):
    np.testing.assert_allclose(a.numpy(), b.numpy())

# 无状态层不会改变权重的顺序或数量，因此即便存在额外的/缺失的无状态层，模型也可以具有兼容架构
inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

inputs = keras.Input(shape=(784,), name="digits")
x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
# Add a dropout layer, which does not contain any weights.
x = keras.layers.Dropout(0.5)(x)
outputs = keras.layers.Dense(10, name="predictions")(x)
functional_model_with_dropout = keras.Model(inputs=inputs, outputs=outputs, name="3_layer_mlp")

functional_model_with_dropout.set_weights(functional_model.get_weights())

# 4. 用于将权重保存到磁盘并将其加载回来的 API
'''
可以用以下格式调用 model.save_weights,将权重保存到磁盘:TensorFlow 检查点和HDF5

model.save_weights 的默认格式是 TensorFlow 检查点。可以通过以下两种方式指定保存格式：
save_format 参数：将值设置为 save_format="tf" 或 save_format="h5"。
path 参数：如果路径以 .h5 或 .hdf5 结束，则使用 HDF5 格式。除非设置了 save_format,否则对于其他后缀,将使用 TensorFlow 检查点格式。
'''
# Runnable example
sequential_model = keras.Sequential(
    [
        keras.Input(shape=(784,), name="digits"),
        keras.layers.Dense(64, activation="relu", name="dense_1"),
        keras.layers.Dense(64, activation="relu", name="dense_2"),
        keras.layers.Dense(10, name="predictions"),
    ]
)

sequential_model.save_weights("weights.h5")
sequential_model.load_weights("weights.h5")

sequential_model.save_weights("ckpt")
load_status = sequential_model.load_weights("ckpt")

# `assert_consumed` can be used as validation that all variable values have been
# restored from the checkpoint. See `tf.train.Checkpoint.restore` for other
# methods in the Status object.
load_status.assert_consumed()