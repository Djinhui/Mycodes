import tensorflow as tf
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

data = datasets.load_iris()

# 可以正常运行
x_data = data.data
y_data = data.target

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
np.random.seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1,seed=1))

lr = 0.1
train_loss_results = []
test_acc = []
epochs = 500
loss_all = 0 
step = 4 # x_train.shape[0]/batch_size

m_w = 0
m_b = 0
beta = 0.9
beta1 = 0.9
beta2 = 0.999
delta_w = 0
delta_b = 0
global_step = 0

v_w = 0
v_b = 0
for epoch in range(epochs):
    for step, (x_train, y_train) in enumerate(train_db):
        global_step += 1
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y_ - y))
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1])
        
        # sgd
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

        # sgd with momentum
        m_w = beta * m_w + (1 - beta) * grads[0]
        m_b = beta * m_b + (1 - beta) * grads[1]
        w1.assign_sub(lr * m_w)
        b1.assign_sub(lr * m_b)

        # adagrad
        v_w += tf.square(grads[0])
        v_b += tf.square(grads[1])
        w1.assign_sub(lr * grads[0] / tf.sqrt(v_w + 1e-8))
        b1.assign_sub(lr * grads[1] / tf.sqrt(v_b + 1e-8))

        # rmsprop
        v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
        v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
        w1.assign_sub(lr * grads[0] / tf.sqrt(v_w + 1e-8))
        b1.assign_sub(lr * grads[1] / tf.sqrt(v_b + 1e-8))

        # adam
        m_w = beta1 * m_w + (1 - beta1) * grads[0]
        m_b = beta1 * m_b + (1 - beta1) * grads[1]
        v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
        v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])
        m_w_correction = m_w / (1 - beta1 ** (global_step + 1))
        m_b_correction = m_b / (1 - beta1 ** (global_step + 1))
        v_w_correction = v_w / (1 - beta2 ** (global_step + 1))
        v_b_correction = v_b / (1 - beta2 ** (global_step + 1))

        w1.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction + 1e-8))
        b1.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction + 1e-8))


    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(loss_all / step))
    train_loss_results.append(loss_all / step)
    loss_all = 0 # loss_all归零，记录下一个epoch的loss

    total_correct, total_num = 0,0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_num += x_test.shape[0]

    acc = total_correct / total_num
    test_acc.append(acc)
    print('Test Accuracy:', acc)
    print()







