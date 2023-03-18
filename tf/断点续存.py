import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
# np.set_printoptions(threshold=np.inf)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
        ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorivcal_accuracy'])

checkpoint_save_path = './chekpoint/mnist_cnn.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,save_weights_only=True,save_best_only=True)
history = model.fit(x_train, y_train, epochs=5, batch_size=64, callbacks=[cp_callback])

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# 给图识物
predNum = int(input("请输入预测的数量："))
for i in range(predNum):
    image_path = input("请输入图片路径：")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 1

    img_arr = img_arr.reshape(1, 28, 28)
    x_predict = model.predict(img_arr)
    pred = tf.argmax(x_predict, axis=1)
    print('the predict result is %d' % pred)


acc = history.history('sparse_categorivcal_accuracy')
val_acc = history.history('val_sparse_categorivcal_accuracy')
loss = history.history('loss')
val_loss = history.history('val_loss')

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


                                                        
# 给图识物
predNum = int(input("请输入预测的数量："))
for i in range(predNum):
    image_path = input("请输入图片路径：")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_arr = np.array(img.convert('L'))

    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 0
            else:
                img_arr[i][j] = 1

    img_arr = img_arr.reshape(1, 28, 28)
    x_predict = model.predict(img_arr)
    pred = tf.argmax(x_predict, axis=1)
    print('the predict result is %d' % pred)


