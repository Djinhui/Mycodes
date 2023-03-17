import tensorflow as tf
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

TRUE_W = 3.0
TRUE_B = 2.0
NUM_EAMPLES = 201

x = tf.linspace(-2, 2, NUM_EAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
    return x * TRUE_W + TRUE_B

noise = tf.random.normal(shape=[NUM_EAMPLES])
y = f(x) + noise

# 1.  使用t.Module完成

class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.w * x + self.b

model = MyModel()
print('Variables:', model.variables)
assert model(3.0).numpy() == 15.0

def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(model, x, y, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(y, model(x))

    dw,db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

weights = []
biases = []
epochs = range(10)

def report(model, loss):
    return f'W={model.w.numpy():1.2f}, b={model.b.numpy():1.2f}, loss={loss:2.2f}'

def training_loop(model, x, y):
    for epoch in epochs:
        train(model, x, y, 0.01)

        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss = loss(y, model(x))
        print(f'Epoch {epoch:2d}:')
        print('      ', report(model, current_loss))

current_loss = loss(y, model(x))

print(f"Starting:")
print("    ", report(model, current_loss))

training_loop(model, x, y)

plt.plot(epochs, weights, label='Weights', color=colors[0])
plt.plot(epochs, [TRUE_W] * len(epochs), '--',
         label = "True weight", color=colors[0])

plt.plot(epochs, biases, label='bias', color=colors[1])
plt.plot(epochs, [TRUE_B] * len(epochs), "--",
         label="True bias", color=colors[1])

plt.legend()
plt.show()


# 2. 使用Keras完成
class MyModelKeras(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def call(self, x):
        return self.w * x + self.b

keras_model = MyModelKeras()

training_loop(keras_model, x, y)

keras_model.compile(optimizer=tf.keras.optimizers.SGD(0.01), loss=tf.keras.losses.mean_squared_error)
keras_model.fit(x, y, epochs=10, batch_size=1000)

keras_model.save_weights("my_checkpoint")