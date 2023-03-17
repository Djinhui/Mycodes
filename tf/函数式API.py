import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 函数式 API 可以处理具有非线性拓扑的模型、具有共享层的模型，以及具有多个输入或输出的模型

inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
model.summary()
keras.utils.plot_model(model, 'mymodel.png', show_shape=True)

# 2. 使用函数式 API 构建的模型，其训练、评估和推断的工作方式与 Sequential 模型完全相同
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255


"""
分类 from_logits=True更稳定
没有onehot,没有softmax,loss=SparseCategoricalCrossentropy(from_logits=True)
没有onehot,有softmax,loss=SparseCategoricalCrossentropy(from_logits=False)
有onehot,没有softmax,loss=CategoricalCrossentropy(from_logits=True)
有onehot,有softmax,loss=CategoricalCrossentropy(from_logits=False)
"""
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# 3. 保存 加载
model.save('path_to_model')
del model
model = keras.models.load_model('path_to_model')

# 所有模型均可像层一样调用
encoder_input = keras.Input(shape=(28,28,1), name='img')
x = layers.Conv2D(16,3,activation='relu')(encoder_input)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32,3,activation='relu')(x)
x = layers.Conv2D(16,3,activation='relu')(x)
encoder_output = layers.GlobalAveragePooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

x = layers.Reshape((4,4,1))(encoder_output)
x = layers.Conv2DTranspose(16,3, activation='relu')(x)
x = layers.Conv2DTranspose(32,3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1,3, activaiton='relu')(x)

autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()

# 可以通过在 Input 上或在另一个层的输出上调用任何模型来将其当作层来处理。
# 4. 通过调用模型，您不仅可以重用模型的架构，还可以重用它的权重。

encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name='encoded_img')
x = layers.Reshape((4,4,1))(decoder_input)
x = layers.Conv2DTranspose(16,3, activation='relu')(x)
x = layers.Conv2DTranspose(32,3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16,3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1,3, activaiton='relu')(x)
decoder = keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()

autoencoder_input = keras.Input(shape=(28,28,1), name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()

# 5. 模型集成
def get_model():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)

model1 = get_model()
model2 = get_model()
model3 = get_model()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1,y2,y3])
ensemble_model = keras.Model(inputs, outputs)

# 6. 复杂的计算图拓扑——具有多个输入和输出的模型

num_tags = 12  # Number of unique issue tags
num_words = 10000 # Size of vocabulary obtained when preprocessing text data
num_departments = 4 # Number of departments for predictions

title_input = keras.Input(shape=(None,), name='title') # Variable-length sequence of ints
body_input = keras.Input(shape=(None,), name='body') # Variable-length sequence of ints
tags_input = keras.Input(shape=(num_tags,), name='tags') # Binary vectors of size `num_tags`

# Embed each word in the title into a 64-dimensional vector
title_features = layers.Embedding(num_words, 64)(title_input)
# Embed each word in the text into a 64-dimensional vector
body_features = layers.Embedding(num_words, 64)(body_input)

# Reduce sequence of embedded words in the title into a single 128-dimensional vector
title_features = layers.LSTM(128)(title_features)
# Reduce sequence of embedded words in the body into a single 32-dimensional vector
body_features = layers.LSTM(32)(body_features)

# Merge all available features into a single large vector via concatenation
x = layers.concatenate([title_features, body_features, tags_input])

# Stick a logistic regression for priority prediction on top of the features
priority_pred = layers.Dense(1, name="priority")(x)
# Stick a department classifier on top of the features
department_pred = layers.Dense(num_departments, name="department")(x)

# Instantiate an end-to-end model predicting both priority and department
model = keras.Model(
    inputs=[title_input, body_input, tags_input],
    outputs=[priority_pred, department_pred],
)

model.compile(optimizer=keras.optimizers.RMSprop(1e-3),\
    loss=[
        keras.losses.BinaryCrossentropy(from_logits=True),
        keras.losses.CategoricalCrossentropy(from_logits=True)
    ],
    loss_weights=[1.0,0.2])

# or 由于输出层具有不同的名称，还可以使用对应的层名称指定损失和损失权重
model.compile(optimizer=keras.optimizers.RMSprop(1e-3),\
    loss = {'priority':keras.losses.BinaryCrossentropy(from_logits=True),
    'department':keras.losses.CategoricalCrossentropy(from_logits=True)},
    loss_Weights = {'priority':1.0, 'department':0.2})

# Dummy input data
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype("float32")

# Dummy target data
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))

model.fit(
    {"title": title_data, "body": body_data, "tags": tags_data},
    {"priority": priority_targets, "department": dept_targets},
    epochs=2,
    batch_size=32,
)

# 小ResNet模型
inputs = keras.Input(shape=(32,32,3), name='img')
x = layers.Conv2D(32,3, activation='relu')(inputs)
x = layers.Conv2D(64,3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)

x = layers.Conv2D(64,3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64,3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])

x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
block_3_output = layers.add([x, block_2_output])

x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs, outputs, name='toy_resnet')

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["acc"],
)

model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)

# 6. 共享层：共享层是在同一个模型中多次重用的层实例
# Embedding for 1000 unique words mapped to 128-dimensional vectors
shared_embedding = layers.Embedding(1000, 128)

# Variable-length sequence of integers
text_input_a = keras.Input(shape=(None,), dtype="int32")
text_input_b = keras.Input(shape=(None,), dtype="int32")

# 共享层通常用于对来自相似空间（例如，两个具有相似词汇的不同文本）的输入进行编码
# Reuse the same layer to encode both inputs
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)

# 7. 特征提取：提取和重用层计算图中的节点
vgg19 = tf.keras.applications.VGG19()
features_list = [layer.output for layer in vgg19.layers]
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)

img = np.random.random((1, 224, 224, 3)).astype("float32")
extracted_features = feat_extraction_model(img)

# 8.使用自定义层扩展
class CustomDense(layers.Layer):
    def __init__(self,units=32):
        super(CustomDense, self).__init__()
        self.units = units

    # 更加快捷的方式为层添加权重：add_weight() 方法
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),\
            initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units, ),\
            initializer='random_normal', trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    # 为了在您的自定义层中支持序列化，请定义一个get_config方法，该方法返回该层实例的构造函数参数
    def get_config(self):
        config = super(CustomDense,self).get_config()
        config.update({'units':self.units})
        return config

inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config, custom_objects={"CustomDense": CustomDense})

# 9. 混合使用Sequential 模型、函数式模型、子类化模型
units = 32
timesteps = 10
input_dim = 5

# Define a Functional model
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

class CustomRNN(layers.Layer):
    def __init__(self):
        super(CustomRNN, self).__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation="tanh")
        self.projection_2 = layers.Dense(units=units, activation="tanh")
        # Our previously-defined Functional model
        self.classifier = model

    def call(self, inputs):
        outputs = []
        state = tf.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = tf.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)


rnn_model = CustomRNN()
_ = rnn_model(tf.zeros((1, timesteps, input_dim)))