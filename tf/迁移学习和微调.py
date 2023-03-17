'''
在深度学习情境中，迁移学习最常见的形式是以下工作流：

从之前训练的模型中获取层。
冻结这些层，以避免在后续训练轮次中破坏它们包含的任何信息。
在已冻结层的顶部添加一些新的可训练层。这些层会学习将旧特征转换为对新数据集的预测。
在您的数据集上训练新层。
最后一个可选步骤是微调，包括解冻上面获得的整个模型（或模型的一部分），
然后在新数据上以极低的学习率对该模型进行重新训练。以增量方式使预训练特征适应新数据，有可能实现有意义的改进。
'''
# 层和模型具有三个权重属性：weights，trainable_weights，non_trainable_weights
'''
请勿将 layer.trainable 特性与 layer.__call__() 中的 training 参数
（此参数控制层是在推断模式还是训练模式下运行其前向传递）混淆
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras


layer = keras.layers.BatchNormalization()
layer.build((None, 4))  # Create the weights

print("weights:", len(layer.weights))  # 4
print("trainable_weights:", len(layer.trainable_weights))  # 2
print("non_trainable_weights:", len(layer.non_trainable_weights))  # 2

# Make a model with 2 layers
layer1 = keras.layers.Dense(3, activation="relu")
layer2 = keras.layers.Dense(3, activation="sigmoid")
model = keras.Sequential([keras.Input(shape=(3,)), layer1, layer2])

# Freeze the first layer
layer1.trainable = False

# Keep a copy of the weights of layer1 for later reference
initial_layer1_weights_values = layer1.get_weights()

# Train the model
model.compile(optimizer="adam", loss="mse")
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# Check that the weights of layer1 have not changed during training
final_layer1_weights_values = layer1.get_weights()
np.testing.assert_allclose(
    initial_layer1_weights_values[0], final_layer1_weights_values[0]
)
np.testing.assert_allclose(
    initial_layer1_weights_values[1], final_layer1_weights_values[1]
)

# 2. trainable 特性的递归设置
# 如果在模型或具有子层的任何层上设置 trainable = False，则所有子层也将变为不可训练
inner_model = keras.Sequential(
    [
        keras.Input(shape=(3,)),
        keras.layers.Dense(3, activation="relu"),
        keras.layers.Dense(3, activation="relu"),
    ]
)

model = keras.Sequential(
    [keras.Input(shape=(3,)), inner_model, keras.layers.Dense(3, activation="sigmoid"),]
)

model.trainable = False  # Freeze the outer model

assert inner_model.trainable == False  # All layers in `model` are now frozen
assert inner_model.layers[0].trainable == False  # `trainable` is propagated recursively

# 3. 典型的迁移学习工作流
'''
典型的迁移学习工作流：

实例化一个基础模型并加载预训练权重。
通过设置 trainable = False 冻结基础模型中的所有层。
根据基础模型中一个（或多个）层的输出创建一个新模型。
在您的新数据集上训练新模型。

请注意，另一种更轻量的工作流如下：
实例化一个基础模型并加载预训练权重。
通过该模型运行新的数据集，并记录基础模型中一个（或多个）层的输出。这一过程称为特征提取。
使用该输出作为新的较小模型的输入数据。

第二种工作流有一个关键优势，即您只需在自己的数据上运行一次基础模型，而不是每个训练周期都运行一次。
因此，它的速度更快，开销也更低。

但是，第二种工作流存在一个问题，即它不允许您在训练期间动态修改新模型的输入数据，在进行数据扩充时，
这种修改必不可少。当新数据集的数据太少而无法从头开始训练完整模型时，任务通常会使用迁移学习，
在这种情况下，数据扩充非常重要。因此，在接下来的篇幅中，我们将专注于第一种工作流。
'''

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False)  # Do not include the ImageNet classifier at the top.
base_model.trainable = False

inputs = keras.Input(shape=(150, 150, 3))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

new_dataset = ''
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)

# 4. 微调
# 解冻(base_model.trainable = True)-->重新compile(Adam(1e-5)) --> 微调fit
'''
一旦模型在新数据上收敛，您就可以尝试解冻全部或部分基础模型，并以极低的学习率端到端地重新训练整个模型。
这是可选的最后一个步骤，可能给您带来增量式改进。不过，它也可能导致快速过拟合，请牢记这一点。

重要的是，只有在将具有冻结层的模型训练至收敛后，才能执行此步骤。
如果将随机初始化的可训练层与包含预训练特征的可训练层混合使用，
则随机初始化的层将在训练过程中引起非常大的梯度更新，而这将破坏您的预训练特征

在此阶段使用极低的学习率也很重要，因为与第一轮训练相比，您正在一个通常非常小的数据集上训练一个大得多的模型。
因此，如果您应用较大的权重更新，则存在很快过拟合的风险。在这里，您只需要以增量方式重新调整预训练权重。
'''

# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
# 如果您更改任何 trainable 值，请确保在您的模型上再次调用 compile() 以将您的变更考虑在内
model.compile(optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])

# Train end-to-end. Be careful to stop before you overfit!
model.fit(new_dataset, epochs=10, callbacks=..., validation_data=...)

# 5. 使用自定义训练循环进行迁移学习和微调
# Create base model
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)
# Freeze base model
base_model.trainable = False

# Create new model on top.
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for inputs, targets in new_dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

# 6. 端到端示例：基于 Dogs vs. Cats 数据集微调图像分类模型
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print(
    "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
)
print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))

size = (150, 150)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

batch_size = 32
train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)

from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be scaled
# from (0, 255) to a range of (-1., +1.), the rescaling layer
# outputs: `(inputs * scale) + offset`
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.summary() # Trainable params: 2,049， Non-trainable params: 20,861,480

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 20
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary() # Trainable params: 20,809,001， Non-trainable params: 54,528(training=False-->BN层的beta和gamma不在更新)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)