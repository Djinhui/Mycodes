# https://keras.io/examples/timeseries/timeseries_classification_transformer/

import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0] # the first column corresponds to the label
    x = data[:, 1:]
    return x, y.astype(int)


root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"

x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
x_test, y_test = readucr(root_url + "FordA_TEST.tsv")

# 归一化【加载的数据已经处理过】
# 每个时序样本长度500， 每个时序样本0均值方差为1 (Do Normalizer not Standarizer On X_train/X_test)

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

num_classes = len(np.unique(y_train))

#shuffle
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

# Standardize the labels to positive integers. The expected labels will then be 0 and 1
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)  # (None, 500, 1)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)  # (None, 500, 1)
    res = x + inputs  # (None, 500, 1)
    
    # Feed Forward Part
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(res)  # (None, 500, 4)
    x = keras.layers.Dropout(dropout)(x)  # (None, 500, 4)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)  # (None, 500, 1)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res  # (None, 500, 1)

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units,dropout=0,mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)  # (None, 500, 1)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout) # (None, 500, 1)

    x = keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)   # (None, 500)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation='relu')(x) # (None, 128)
        x = keras.layers.Dropout(mlp_dropout)(x)

    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)  # (None, 2)
    return keras.models.Model(inputs, outputs)


input_shape = x_train.shape[1:]

model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4,
                    mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)
model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=64,
    callbacks=callbacks,
)

model.evaluate(x_test, y_test, verbose=1)