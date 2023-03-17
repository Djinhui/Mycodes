'''
您在进行监督学习时可以使用 fit()，一切都可以顺利完成。
需要从头开始编写自己的训练循环时，您可以使用 GradientTape 并控制每个微小的细节。

但如果您需要自定义训练算法，又想从 fit() 的便捷功能（例如回调、内置分布支持或步骤融合）中受益，那么该怎么做？
需要自定义 fit() 的功能时，您应重写 Model 类的train_step()函数。此函数是 fit() 会针对每批次数据调用的函数。
然后，您将能够像往常一样调用 fit()，它将运行您自己的学习算法。

此模式不会妨碍您使用函数式 API 构建模型。
无论是构建 Sequential 模型、函数式 API 模型还是子类模型，均可采用这种模式。

'''
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 
# compile()中传入了loss和metric,则仅重新train_step(self, data)即可
# 并通过 self.compiled_loss 计算损失，
# 调用 self.compiled_metrics.update_state(y, y_pred) 来更新在 compile() 中传递的指标的状态
# 并在最后从 self.metrics 中查询结果以检索其当前值



class CustomModel(keras.Model):
    def train_step(self, data):
        '''
        如果通过调用 fit(x, y, ...) 传递 Numpy 数组，则 data 将为元祖 (x, y)
        如果通过调用 fit(dataset, ...) 传递 tf.data.Dataset，则 data 将为每批次 dataset 产生的数据
        '''
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name:m.result() for m in self.metrics}


# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Just use `fit` as usual
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)


# 2. 仅在compile() 配置优化器
loss_tracker = keras.metrics.Mean(name='loss')
mae_tracker = keras.metrics.MeanAbsoluteError(name='mae')

class CustomModel(keras.Model):
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute our own loss
            loss = keras.losses.mean_squared_error(y, y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        loss_tracker.update_state(loss)
        mae_tracker.update_state(y, y_pred)
        return {'loss':loss_tracker.result(), 'mae':mae_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.

        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker, mae_tracker]

# We don't passs a loss or metrics here.
model.compile(optimizer="adam")

# Just use `fit` as usual -- you can use callbacks, etc.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)

# 3. 支持 sample_weight 和 class_weight
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct and compile an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# You can now use sample_weight argument
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)

# 4. 自定义评估步骤:重新test_step()
class CustomModel(keras.Model):
    def test_step(self, data):
        # Unpack the data
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}


# Construct an instance of CustomModel
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss="mse", metrics=["mae"])

# Evaluate with our custom test_step
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)


# 5. 利用上述全部内容的端到端示例 GAN

# Create the discriminator
discriminator = keras.Sequential([
    keras.Input(shape=(28,28,1)),
    layers.Conv2D(64, 3, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(128, 3, strides=2, padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.GlobalMaxPooling2D(),
    layers.Dense(1) # 二分类
], name='discriminator')

# Create the generator
latent_dim = 128
generator = keras.Sequential([
    keras.Input(shape=(latent_dim,)),
    # We want to generate 128 coefficients to reshape into a 7x7x128 map
    layers.Dense(7*7*128),
    layers.LeakyReLU(alpha=0.2),
    layers.Reshape((7, 7, 128)),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
], name="generator")


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    # 重写compile() 用于两个网络的优化器
    def compile(self, d_optimizer, g_optimier, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimier = g_optimier
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # 1. Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # 2. Decode them to fake images
        generated_images = self.generator(random_latent_vectors)
        # 3. Combine the with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)
        # 4. Assemble labels discriminating real from fake images. 1 for fake,  0 for real
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # 5. Add random noise to the labels  -- Important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # 6. Train the discriminator and update the weights of the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # 7. Sample random points in the latent space again
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # 8. Assemble labels that say !!!!"all real images"!!!!:0, actual they are 1 for fake
        misleading_labels = tf.zeros((batch_size, 1))

        # 9. Train the generator (note that we should *not* update the weights of the discriminator)
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'d_loss':d_loss, 'g_loss':g_loss}

# Prepare the dataset. We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

# To limit the execution time, we only train on 100 batches. You can train on
# the entire dataset. You will need about 20 epochs to get nice results.
gan.fit(dataset.take(100), epochs=1)




    
