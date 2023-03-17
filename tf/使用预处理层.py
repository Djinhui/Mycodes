import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. 文本预处理层
'''
在tensorflow中完成文本数据预处理的常用方案有两种
第一种是利用tf.keras.preprocessing中的Tokenizer词典构建工具和tf.keras.utils.Sequence构建文本数据生成器管道。
第二种是使用tf.data.Dataset搭配tf.keras.layers.experimental.preprocessing.TextVectorization预处理层。
'''
# tf.keras.layers.experimental.preprocessing.TextVectorization
# tf.keras.preprocessing.sequence import pad_sequences
# tf.keras.preprocessing.text import Tokenizer

# tf.keras.layers.TextVectorization: turns raw strings into an encoded representation that 
# can be read by an Embedding layer or Dense layer

# 2. 数值特征预处理层
# tf.keras.layers.Normalization: performs feature-wise normalize of input features.
# tf.keras.layers.Discretization: turns continuous numerical features into integer categorical features.

# 3. 类别特征预处理层
# tf.keras.layers.CategoryEncoding: turns integer categorical features into one-hot, multi-hot, or count dense representations.
# tf.keras.layers.Hashing: performs categorical feature hashing, also known as the "hashing trick".
# tf.keras.layers.StringLookup: turns string categorical values an encoded representation that can be read by an Embedding layer or Dense layer.
# tf.keras.layers.IntegerLookup: turns integer categorical values into an encoded representation that can be read by an Embedding layer or Dense layer.

# 4. 图片预处理层
# tf.keras.layers.Resizing: resizes a batch of images to a target size.
# tf.keras.layers.Rescaling: rescales and offsets the values of a batch of image (e.g. go from inputs in the [0, 255] range to inputs in the [0, 1] range.
# tf.keras.layers.CenterCrop: returns a center crop of a batch of images.

# 5. 图片数据增强-训练期间
# tf.keras.layers.RandomCrop
# tf.keras.layers.RandomFlip
# tf.keras.layers.RandomTranslation
# tf.keras.layers.RandomRotation
# tf.keras.layers.RandomZoom
# tf.keras.layers.RandomHeight
# tf.keras.layers.RandomWidth
# tf.keras.layers.RandomContrast

# 6. adapt() 方法
# Some preprocessing layers have an internal state that can be computed based on a sample of the training data
data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7],])
layer = layers.Normalization()  # 类似scaler = sklearn.precrocessing.Nornalizer()
layer.adapt(data)               # 类似scaler.fit(data)
normalized_data = layer(data)   # 类似scaler.transform(data)

print("Features mean: %.2f" % (normalized_data.numpy().mean()))
print("Features std: %.2f" % (normalized_data.numpy().std()))

data = [
    "ξεῖν᾽, ἦ τοι μὲν ὄνειροι ἀμήχανοι ἀκριτόμυθοι",
    "γίγνοντ᾽, οὐδέ τι πάντα τελείεται ἀνθρώποισι.",
    "δοιαὶ γάρ τε πύλαι ἀμενηνῶν εἰσὶν ὀνείρων:",
    "αἱ μὲν γὰρ κεράεσσι τετεύχαται, αἱ δ᾽ ἐλέφαντι:",
    "τῶν οἳ μέν κ᾽ ἔλθωσι διὰ πριστοῦ ἐλέφαντος,",
    "οἵ ῥ᾽ ἐλεφαίρονται, ἔπε᾽ ἀκράαντα φέροντες:",
    "οἱ δὲ διὰ ξεστῶν κεράων ἔλθωσι θύραζε,",
    "οἵ ῥ᾽ ἔτυμα κραίνουσι, βροτῶν ὅτε κέν τις ἴδηται.",
]
layer = layers.TextVectorization()
layer.adapt(data)
vectorized_text = layer(data)
print(vectorized_text) # shape=(8, 9)

# 示例1
# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
x_train = x_train.reshape((len(x_train), -1))
input_shape = x_train.shape[1:]
classes = 10

# Create a Normalization layer and set its internal state using the training data
normalizer = layers.Normalization()
normalizer.adapt(x_train)

# Create a model that include the normalization layer
inputs = keras.Input(shape=input_shape)
x = normalizer(inputs)
outputs = layers.Dense(classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# Train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
model.fit(x_train, y_train)

# 示例2
# Create a data augmentation stage with horizontal flipping, rotations, zooms
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# Load some data
(x_train, y_train), _ = keras.datasets.cifar10.load_data()
input_shape = x_train.shape[1:]
classes = 10

# Create a tf.data pipeline of augmented images (and their labels)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(16).map(lambda x, y: (data_augmentation(x), y))


# Create a model and train it on the augmented image data
inputs = keras.Input(shape=input_shape)
x = layers.Rescaling(1.0 / 255)(inputs)  # Rescale inputs
outputs = keras.applications.ResNet50(  # Add the rest of the model
    weights=None, input_shape=input_shape, classes=classes
)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
model.fit(train_dataset, steps_per_epoch=5)

# 示例3
# Define some text data to adapt the layer
adapt_data = tf.constant(
    [
        "The Brain is wider than the Sky",
        "For put them side by side",
        "The one the other will contain",
        "With ease and You beside",
    ]
)
# Instantiate TextVectorization with "tf-idf" output_mode
# (multi-hot with TF-IDF weighting) and ngrams=2 (index all bigrams)
text_vectorizer = layers.TextVectorization(output_mode="tf-idf", ngrams=2)
# Index the bigrams and learn the TF-IDF weights via `adapt()`

with tf.device("CPU"):
    # A bug that prevents this from running on GPU for now.
    text_vectorizer.adapt(adapt_data)

# Try out the layer
print(
    "Encoded text:\n", text_vectorizer(["The Brain is deeper than the sea"]).numpy(),
)

# Create a simple model
inputs = keras.Input(shape=(text_vectorizer.vocabulary_size(),))
outputs = layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

# Create a labeled dataset (which includes unknown tokens)
train_dataset = tf.data.Dataset.from_tensor_slices(
    (["The Brain is deeper than the sea", "for if they are held Blue to Blue"], [1, 0])
)

# Preprocess the string inputs, turning them into int sequences
train_dataset = train_dataset.batch(2).map(lambda x, y: (text_vectorizer(x), y))
# Train the model on the int sequences
print("\nTraining model...")
model.compile(optimizer="rmsprop", loss="mse")
model.fit(train_dataset)

# For inference, you can export a model that accepts strings as input
inputs = keras.Input(shape=(1,), dtype="string")
x = text_vectorizer(inputs)
outputs = model(x)
end_to_end_model = keras.Model(inputs, outputs)

# Call the end-to-end model on test data (which includes unknown tokens)
print("\nCalling end-to-end model on test string...")
test_data = tf.constant(["The one the other will absorb"])
test_output = end_to_end_model(test_data)
print("Model output:", test_output)