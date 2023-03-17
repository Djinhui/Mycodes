import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

'''
class CIFAR10Sequence(keras.utils.Sequence):
    def __init__(self, filenames, labels,batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx*self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(imread(filename), (200, 200))for filename in batch_x]), \
            np.array(batch_y)

filenames = ['']
labels = ['']
batch_size = 64
sequence = CIFAR10Sequence(filenames, labels, batch_size)
model.fit(sequence, epochs=10)
'''
# 一： 构建数据管道
# 1. 从Numpy array 构建数据管道
iris = datasets.load_iris()
ds1 = tf.data.Dataset.from_tensor_slices((iris['data'], iris['target']))
for features, label in ds1.take(5):
    print(features, label)


# 2. 从 Pandas DataFrame构建数据管道
dfiris = pd.DataFrame(iris["data"],columns = iris.feature_names)
ds2 = tf.data.Dataset.from_tensor_slices((dfiris.to_dict('list'), iris['target']))
for features,label in ds2.take(3):
    print(features,label)

# 3. 从Python generator构建数据管道
from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(rescale=1/255.).flow_from_directory(
    './data/caifar2/test', target_size=(32,32), batch_size=32, class_mode='binary'
)

classdict = img_gen.class_indices
def generator():
    for features, label in img_gen:
        yield (features, label)

ds3 = tf.data.Dataset.from_generator(generator, output_types=(tf.float32, tf.int32))
plt.figure(figsize=(6,6)) 
for i,(img,label) in enumerate(ds3.unbatch().take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow(img.numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()

# 4. 从csv文件构建数据管道
ds4 = tf.data.experimental.make_csv_dataset(
    file_pattern=["../../data/titanic/train.csv","../../data/titanic/test.csv"],
    batch_size=3,
    label_name='Suriviced',
    na_value='',
    num_epochs=1,
    ignore_errors=True
)

for data,label in ds4.take(2):
    print(data,label)

# 5. 从文本文件构建数据管道
ds5 = tf.data.TextLineDataset(filenames=["../../data/titanic/train.csv","../../data/titanic/test.csv"]).skip(1) # 去掉header
for line in ds5.take(5):
    print(line)

# 6. 从文件路径构建数据管道
ds6 = tf.data.Dataset.list_files('./data/cifar2/train/*/*.jpg')
for file_path in ds6.take(5):
    print(file_path)

def load_image(img_path, size=(32,32)):
    label = 1 if tf.strings.regex_full_match(img_path, '.*/auto,obile/.*') else 0
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, size)
    return img, label

for i,(img,label) in enumerate(ds6.map(load_image).take(2)):
    plt.figure(i)
    plt.imshow((img/255.0).numpy())
    plt.title("label = %d"%label)
    plt.xticks([])
    plt.yticks([])

# 7. 从tfrecords文件构建数据管道
import os
import numpy as np

# inpath：原始数据路径 outpath:TFRecord文件输出路径
def create_tfrecords(inpath,outpath): 
    writer = tf.io.TFRecordWriter(outpath)
    dirs = os.listdir(inpath)
    for index, name in enumerate(dirs):
        class_path = inpath +"/"+ name+"/"
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = tf.io.read_file(img_path)
            #img = tf.image.decode_image(img)
            #img = tf.image.encode_jpeg(img) #统一成jpeg格式压缩
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.numpy()]))
               }))
            writer.write(example.SerializeToString())
    writer.close()

create_tfrecords("../../data/cifar2/test/","../../data/cifar2_test.tfrecords/")

from matplotlib import pyplot as plt 

def parse_example(proto):
    description ={ 'img_raw' : tf.io.FixedLenFeature([], tf.string),
                   'label': tf.io.FixedLenFeature([], tf.int64)} 
    example = tf.io.parse_single_example(proto, description)
    img = tf.image.decode_jpeg(example["img_raw"])   #注意此处为jpeg格式
    img = tf.image.resize(img, (32,32))
    label = example["label"]
    return(img,label)

ds7 = tf.data.TFRecordDataset("../../data/cifar2_test.tfrecords").map(parse_example).shuffle(3000)

plt.figure(figsize=(6,6)) 
for i,(img,label) in enumerate(ds7.take(9)):
    ax=plt.subplot(3,3,i+1)
    ax.imshow((img/255.0).numpy())
    ax.set_title("label = %d"%label)
    ax.set_xticks([])
    ax.set_yticks([]) 
plt.show()

# 二 应用数据转换
'''
Dataset数据结构应用非常灵活,因为它本质上是一个Sequece序列,其每个元素可以是各种类型,例如可以是张量,列表,字典,
也可以是Dataset。

Dataset包含了非常丰富的数据转换功能。

map: 将转换函数映射到数据集每一个元素。
flat_map: 将转换函数映射到数据集的每一个元素,并将嵌套的Dataset压平。
interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。
filter: 过滤掉某些元素。
zip: 将两个长度相同的Dataset横向铰合。
concatenate: 将两个Dataset纵向连接。
reduce: 执行归并操作。
batch : 构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。
padded_batch: 构建批次,类似batch, 但可以填充到相同的形状。
window :构建滑动窗口,返回Dataset of Dataset.
shuffle: 数据顺序洗牌。
repeat: 重复数据若干次，不带参数时，重复无数次。
shard: 采样，从某个位置开始隔固定距离采样一个元素。
take: 采样，从开始位置取前几个元素。
'''
# map:将转换函数映射到数据集每一个元素
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello China","hello Beijing"])
ds_map = ds.map(lambda x:tf.strings.split(x," "))
for x in ds_map:
    print(x)

# flat_map:将转换函数映射到数据集的每一个元素，并将嵌套的Dataset压平。
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello China","hello Beijing"])
ds_flatmap = ds.flat_map(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
for x in ds_flatmap:
    print(x)

# interleave: 效果类似flat_map,但可以将不同来源的数据夹在一起。
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello China","hello Beijing"])
ds_interleave = ds.interleave(lambda x:tf.data.Dataset.from_tensor_slices(tf.strings.split(x," ")))
for x in ds_interleave:
    print(x)

# filter:过滤掉某些元素。
ds = tf.data.Dataset.from_tensor_slices(["hello world","hello China","hello Beijing"])
# 找出含有字母a或B的元素
ds_filter = ds.filter(lambda x: tf.strings.regex_full_match(x, ".*[a|B].*"))
for x in ds_filter:
    print(x)

# zip:将两个长度相同的Dataset横向铰合。
ds1 = tf.data.Dataset.range(0,3)
ds2 = tf.data.Dataset.range(3,6)
ds3 = tf.data.Dataset.range(6,9)
ds_zip = tf.data.Dataset.zip((ds1,ds2,ds3))
for x,y,z in ds_zip:
    print(x.numpy(),y.numpy(),z.numpy())

# condatenate:将两个Dataset纵向连接。
ds1 = tf.data.Dataset.range(0,3)
ds2 = tf.data.Dataset.range(3,6)
ds_concat = tf.data.Dataset.concatenate(ds1,ds2)
for x in ds_concat:
    print(x)

# reduce:执行归并操作。
ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5.0])
result = ds.reduce(0.0,lambda x,y:tf.add(x,y))
result # 15.0

# batch:构建批次，每次放一个批次。比原始数据增加一个维度。 其逆操作为unbatch。 
ds = tf.data.Dataset.range(12)
ds_batch = ds.batch(4)
for x in ds_batch:
    print(x)

# padded_batch:构建批次，类似batch, 但可以填充到相同的形状。
elements = [[1, 2],[3, 4, 5],[6, 7],[8]]
ds = tf.data.Dataset.from_generator(lambda: iter(elements), tf.int32)
ds_padded_batch = ds.padded_batch(2,padded_shapes = [4,])
for x in ds_padded_batch:
    print(x)    

# window:构建滑动窗口，返回Dataset of Dataset.
ds = tf.data.Dataset.range(12)
# window返回的是Dataset of Dataset,可以用flat_map压平
ds_window = ds.window(3, shift=1).flat_map(lambda x: x.batch(3,drop_remainder=True)) 
for x in ds_window:
    print(x)

# shuffle:数据顺序洗牌。
ds = tf.data.Dataset.range(12)
ds_shuffle = ds.shuffle(buffer_size = 5)
for x in ds_shuffle:
    print(x)

# repeat:重复数据若干次，不带参数时，重复无数次。
ds = tf.data.Dataset.range(3)
ds_repeat = ds.repeat(3)
for x in ds_repeat:
    print(x)

# shard:采样，从某个位置开始隔固定距离采样一个元素。
ds = tf.data.Dataset.range(12)
ds_shard = ds.shard(3,index = 1)
for x in ds_shard:
    print(x)

# take:采样，从开始位置取前几个元素。
ds = tf.data.Dataset.range(12)
ds_take = ds.take(3)
list(ds_take.as_numpy_iterator())


# 三 提升管道性能
'''
1使用 prefetch 方法让数据准备和参数迭代两个过程相互并行。
2使用 interleave 方法可以让数据读取过程多进程执行,并将不同来源数据夹在一起。
3使用 map 时设置num_parallel_calls 让数据转换过程多进程执行。
4使用 cache 方法让数据在第一个epoch后缓存到内存中,仅限于数据集不大情形。
5使用 map转换时先batch, 然后采用向量化的转换方法对每个batch进行转换。
'''

# 模拟数据准备
def generator():
    for i in range(10):
        #假设每次准备数据需要2s
        # time.sleep(2) 
        yield i 

ds = tf.data.Dataset.from_generator(generator,output_types = (tf.int32)).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

ds_files = tf.data.Dataset.list_files("../../data/titanic/*.csv")
ds = ds_files.flat_map(lambda x:tf.data.TextLineDataset(x).skip(1))
ds_files = tf.data.Dataset.list_files("../../data/titanic/*.csv")
ds = ds_files.interleave(lambda x:tf.data.TextLineDataset(x).skip(1))

ds = tf.data.Dataset.list_files("../../data/cifar2/train/*/*.jpg")
ds_map_parallel = ds.map(load_image,num_parallel_calls = tf.data.experimental.AUTOTUNE)

# 使用 cache 方法让数据在第一个epoch后缓存到内存中，仅限于数据集不大情形。
ds = tf.data.Dataset.from_generator(generator,output_types = (tf.int32)).cache()

# 先map后batch
ds = tf.data.Dataset.range(100000)
ds_map_batch = ds.map(lambda x:x**2).batch(20)

# 先batch后map
ds = tf.data.Dataset.range(100000)
ds_batch_map = ds.batch(20).map(lambda x:x**2)