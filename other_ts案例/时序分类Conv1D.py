# https://keras.io/examples/timeseries/timeseries_classification_from_scratch/
# The problem is a balanced binary classification task.
# 相关案例see at 《轴承故障预测.ipynb》 特征提取+ML, 子序列Conv1D


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


classes = np.unique(np.concatenate((y_train, y_test), axis=0))
plt.figure()
for c in classes:
    c_x_train = x_train[y_train==c]
    plt.plot(c_x_train[0], label='class ' + str(c))
plt.legend(loc='best')
plt.show()
plt.close()

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

# Fully Convolutional Neural Network
# 超参数是调优后 KerasTuner
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(model, show_shapes=True)


epochs = 500
batch_size = 32
callbacks = [
    keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=0.0001),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=1)
]

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)

model = keras.models.load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc) # 96.8
print("Test loss", test_loss)


metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()