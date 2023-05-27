# https://tensorflow.google.cn/text/guide/word_embeddings

# ======How to represent text as numbers?======

# 1. OneHot encoding
# A one-hot encoded vector is sparse (meaning, most indices are zero).
# Imagine you have 10,000 words in the vocabulary( unique words). To one-hot encode each word, 
# you would create a vector where 99.99% of the elements are zero.

# 2. Encode each word with a unique number
# The integer-encoding is arbitrary (it does not capture any relationship between words).
# An integer-encoding can be challenging for a model to interpret.

# Word Embeddings: use an efficient, dense representation in which similar words have a similar encoding


import io
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir) # ['test', 'imdb.vocab', 'README', 'train', 'imdbEr.txt']

train_dir = os.path.join(dataset_dir, 'train')
os.listdir(train_dir) # ['pos','neg',...]
remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)

for text_batch, label_batch in train_ds.take(1):
    for i in range(5):
        print(label_batch[i].numpy(), text_batch.numpy()[i])

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

vocab_size = 10000
sequence_length = 100

# Use the text vectorization layer to normalize, split, and map strings to integers.
vectorize_layer = TextVectorization(standrdize=custom_standardization, max_tokens=vocab_size,output_mode='int',
                                    output_sequence_length=sequence_length)

# Make a text-only dataset (no labels) and call adapt to build the vocabulary.
text_ds = train_ds.map(lambda x, y:x)
vectorize_layer.adapt(text_ds)

embedding_dim = 16
model = Sequential([
    vectorize_layer, # (None, 100) 
    Embedding(vocab_size, embedding_dim, name='embedding'), # (None, 100, 16) 参数量 vocab*emd_dim 160000
    GlobalAveragePooling1D(),  # (None, 16) 
    Dense(16, activation='relu'),
    Dense(1)
])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[tensorboard_callback])

#docs_infra: no_execute
# %load_ext tensorboard
# %tensorboard --logdir logs

# 可视化
# Retrieve the trained word embeddings and save them to disk
# The weights matrix is of shape (vocab_size, embedding_dimension)
weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vector.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for idx, word in enumerate(vocab):
    if idx == 0:
        continue
    vec = weights[idx]
    out_v.write('\t'.join([str(x) for x in vec]) + '\n')
    out_m.write(word + '\n')

out_v.close()
out_m.close()

# Open the Embedding Projector(http://projector.tensorflow.org/) (this can also run in a local TensorBoard instance).

# Click on "Load data".

# Upload the two files you created above: vecs.tsv and meta.tsv.

