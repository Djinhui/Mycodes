import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Define Sequential model with 3 layers use list
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"), # no specifying the input shape in advance 
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
) # No weights at this stage!

# At this point, you can't do this:
# model.weights

# You also can't do this:
# model.summary()

# It creates its weights the first time it is called on an inpu
# Call model on a test input
x = tf.ones((3, 3))
y1 = model(x)

# now can do this
model.summary()

model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1",input_shape=(4,)), # specifying the input shape in advance 
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
) 

layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")
y2 = layer3(layer2(layer1(x)))

model.layers
model.pop()
len(model.layers)

# 2. Define Sequential model with 3 layers use add()
model = keras.Sequential(name="my_sequential")
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation="relu", name="layer1"))
model.add(layers.Dense(3, activation="relu", name="layer2"))
model.add(layers.Dense(4, name="layer3"))
# or 
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))

model.summary()

# 3. 特征提取 with a Sequential model
initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu", name="my_intermediate_layer"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)

feature_extractor1 = keras.Model(inputs=initial_model.input, \
    outputs=[layer.output for layer in initial_model.layers])

feature_extractor2 = keras.Model(inputs=initial_model.input, \
    outputs=initial_model.get_layer(name='my_intermediate_layer').output)

x = tf.ones((1, 250,250,3))
features = feature_extractor1(x)

# 4. 迁移学习with a Sequential model
# 4.1 you have a Sequential model, and you want to freeze all layers except the last one
model = keras.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])
# Presumably you would want to first load pre-trained weights.
model.load_weights(...)

# Freeze all layers except the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Recompile and train (this will only update the weights of the last layer).
model.compile(...)
model.fit(...)

# 4.2 stack a pre-trained model and some freshly initialized classification layers

# Load a convolutional base with pre-trained weights
base_model = keras.applications.ResNet50(weights="imagenet", include_top=False)
# Freeze the base model
base_model.trainable = False
# Use a Sequential model to add a trainanle classifier on top
model = keras.Sequential([base_model, layers.Dense(100, activation="softmax")])

# Compile & train
model.compile(...)
model.fit(...)


