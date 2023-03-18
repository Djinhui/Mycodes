import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt
import os

input_word = 'abcde'
w_to_id = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4}
id_to_onehot = {0:[1.,0.,0.,0.,0.], 1:[0.,1.,0.,0.,0.], 2:[0.,0.,1.,0.,0.],
                 3:[0.,0.,0.,1.,0.], 4:[0.,0.,0.,0.,1.]}

x_train = [[
    id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']]],
    [id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']]],
    [id_to_onehot[w_to_id['c']], id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']]],
    [id_to_onehot[w_to_id['d']], id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']]],
    [id_to_onehot[w_to_id['e']], id_to_onehot[w_to_id['a']], id_to_onehot[w_to_id['b']], id_to_onehot[w_to_id['c']]]]

y_train = [w_to_id['e'], w_to_id['a'], w_to_id['b'], w_to_id['c'], w_to_id['d']]

np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)
tf.random.set_seed(1)


x_train = np.reshape(x_train, (len(x_train),4,5))
y_train = np.array(y_train)

model = tf.keras.models.Sequential([
    SimpleRNN(3),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
checkpoint_save_path = './checkpoint/SimpleRNN.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True, monitor='loss')

history = model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=[cp_callback])

file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

# acc = history.history('sparse_categorivcal_accuracy')
# val_acc = history.history('val_sparse_categorivcal_accuracy')
# loss = history.history('loss')
# val_loss = history.history('val_loss')

# plt.subplot(1, 2, 1)
# plt.plot(acc, label='Training Accuracy')
# plt.plot(val_acc, label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

preNum = int(input("Enter the number of test alphabet: "))
for i in range(preNum):
    alphabet1 = input("Enter the alphabet: ")
    alphabet = [id_to_onehot[w_to_id[a]] for a in alphabet1]
    alphabet = np.reshape(alphabet,(1,4,5))
    pred = model.predict(alphabet)
    pred = tf.argmax(pred, axis=1)
    pred = int(pred)
    tf.print(alphabet1 + '->' + input_word[pred])




