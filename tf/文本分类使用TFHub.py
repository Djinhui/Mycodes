import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

'''使用文本嵌入向量'''
# Split the training set into 60% and 40% to end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch # <tf.Tensor: shape=(10,), dtype=string, numpy=array([b'This was an absolutely...', b'The film is based..',b'',...b''])
train_labels_batch # <tf.Tensor: shape=(10,), dtype=int64, numpy=array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0])>

# 使用来自 TensorFlow Hub 的 预训练文本嵌入向量模型，名称为 google/nnlm-en-dim50/2
# 请注意无论输入文本的长度如何，嵌入（embeddings）输出的形状都是：(num_examples, embedding_dimension)
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3]) # <tf.Tensor: shape=(3, 50), dtype=float32, numpy=array([[...]])

model = tf.keras.Sequential()
model.add(hub_layer)  # Output Shape (None, 16)
model.add(tf.keras.layers.Dense(16, activation='relu'))  # Output Shape (None, 16)
model.add(tf.keras.layers.Dense(1))  # Output Shape (None, 1)
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=10,
                    validation_data=validation_data.batch(512),
                    verbose=1)

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))