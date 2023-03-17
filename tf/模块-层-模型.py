import tensorflow as tf

########### 1. 在TensorFlow中定义模型和层 ###################

# TensorFlow 中，层和模型的大多数高级实现（例如 Keras 或 Sonnet）都在以下同一个基础类上构建：tf.Module。
# tf.Module 是 tf.keras.layers.Layer 和 tf.keras.Model 的基类
class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(1.0, name='train_me')
        self.non_trainable_varibale = tf.Variable(0.0, trainable=False, name='non_trainable_variable')

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_varibale

simple_module = SimpleModule(name='simple')
simple_module(tf.constant(2.0))

# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)

class MyDense(tf.Module):
    def __init__(self,in_features, out_features,name=None):
        super().__init__(name=name)
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
    
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class MyModel(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.dense1 = MyDense(2, 3, name='dense1')
        self.dense2 = MyDense(3, 1, name='dense2')
    
    def __call__(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        return y

model = MyModel(name='model')
model(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
print('Submodules:', model.submodules)

# 等待创建权重变量：将变量创建推迟到第一次使用特定输入形状调用模块时，您将无需预先指定输入大小
class FlecibleDenseModule(tf.Module):
    # Note:No need for 'in_features'
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, X):
        # Create variables on first call
        if not self.is_built:
            self.w = tf.Variable(tf.random.normal([X.shape[1], self.out_features]), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True
        # Compute output
        y = tf.matmul(X, self.w) + self.b
        return tf.nn.relu(y)


class MyModel2(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.dense1 = FlecibleDenseModule(2)
        self.dense2 = FlecibleDenseModule(3)
    
    def __call__(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        return y

########### 2. keras模型和层 ###################

'''
如果自定义模型层没有需要被训练的参数,一般推荐使用Lamda层实现。
如果自定义模型层有需要被训练的参数,则可以通过对Layer基类子类化实现
'''
mypower = tf.keras.layers.Lambda(lambda x:tf.math.pow(x,2))
mypower(tf.range(5))
# tf.keras.layers.Layer 是所有 Keras 层的基类，它继承自 tf.Module
# 只需换出父项，然后将 __call__ 更改为 call 即可将模块转换为 Keras 层
class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base keras layer arguments
    def __init__(self, in_features, out_Features, **kwargs):
        super().__init__(**kwargs)

        # this step can move to build() step
        self.w = tf.Variable(tf.random.normal([in_features, out_Features]), name='w')
        self.b = tf.Variable(tf.zeros([out_Features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

simple_layer = MyDense(name='simple', in_features=3, out_Features=2)
simple_layer([[2.0,2.0,2.0]])

# build 仅被调用一次，而且是使用输入的形状调用的。它通常用于创建变量（权重)
class FlexibleDense(tf.keras.layers.Layer):
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1], self.out_features]), name='w')
        self.b = tf.Variable(tf.zeros([self.out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)

# Call it, with predictably random results
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))
try: # 由于仅调用一次 build，因此如果输入形状与层的变量不兼容，输入将被拒绝。
  print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
except tf.errors.InvalidArgumentError as e:
  print("Failed:", e)

# Keras 还提供了称为 tf.keras.Model 的全功能模型类。它继承自 tf.keras.layers.Layer
# Keras 模型支持以同样的方式使用、嵌套和保存。Keras 模型还具有额外的功能，这使它们可以轻松训练、评估、加载、保存，甚至在多台机器上进行训练
class MySquentialModel(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = FlexibleDense(out_features=3)
        self.dense2 = FlexibleDense(out_features=3)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x

my_sequential_model  = MySquentialModel(name='the_model')

inputs = tf.keras.Input(shape=[3,])
x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)
my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)

my_sequential_model.save("exname_of_file")
reconstructed_model = tf.keras.models.load_model("exname_of_file")