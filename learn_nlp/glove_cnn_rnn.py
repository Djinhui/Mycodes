# 《自然语言处理实战：从入门到项目实践》CH06 https://github.com/practical-nlp/practical-nlp-code

# 1. 意图检测、意图识别-----本质上是有监督多分类任务
# 1. CNN and RNN for the task of intent detection on the ATIS dataset
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Input, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, Bidirectional,MaxPooling1D
from tensorflow.keras.initializers import Constant
from sklearn.preprocessing import LabelEncoder


def get_data(filename):
    df = pd.read_csv(filename,delim_whitespace=True,names=['word','label'])
    beg_indices = list(df[df['word'] == 'BOS'].index)+[df.shape[0]]
    sents,labels,intents = [],[],[]
    for i in range(len(beg_indices[:-1])):
        sents.append(df[beg_indices[i]+1:beg_indices[i+1]-1]['word'].values)
        labels.append(df[beg_indices[i]+1:beg_indices[i+1]-1]['label'].values)
        intents.append(df.loc[beg_indices[i+1]-1]['label'])    
    return np.array(sents, dtype=object), np.array(labels, dtype=object), np.array(intents, dtype=object)

def get_data2(filename):
    with open(filename) as f:
        contents = f.read()
    sents,labels,intents = [],[],[]
    for line in contents.strip().split('\n'):
        words,labs = [i.split(' ') for i in line.split('\t')]
        sents.append(words[1:-1])
        labels.append(labs[1:-1])
        intents.append(labs[-1])
    return np.array(sents, dtype=object), np.array(labels, dtype=object), np.array(intents, dtype=object)

read_method = {'Data/data2/atis-2.dev.w-intent.iob':get_data,
               'Data/data2/atis.train.w-intent.iob':get_data2,
               'Data/data2/atis.test.w-intent.iob':get_data,
              'Data/data2/atis-2.train.w-intent.iob':get_data2}

def fetch_data(fname):
    func = read_method[fname]
    return func(fname)

sents,labels,intents = fetch_data('Data/data2/atis.train.w-intent.iob')
'''
sents[0]
array(['i', 'would', 'like', 'to', 'find', 'a', 'flight', 'from',
       'charlotte', 'to', 'las', 'vegas', 'that', 'makes', 'a', 'stop',
       'in', 'st.', 'louis'], dtype=object)

labels[0]
array(['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O',
       'B-toloc.city_name', 'I-toloc.city_name', 'O', 'O', 'O', 'O', 'O',
       'B-stoploc.city_name', 'I-stoploc.city_name'], dtype=object)

intents[0]
'atis_flight'
'''

train_sentences = [" ".join(sent) for sent in sents]
train_texts = train_sentences
train_labels = intents.tolist()

vals = []
for i in range(len(train_labels)):
    if '#' in train_labels[i]:
        vals.append(i)
for i in vals[::-1]:
    train_labels.pop(i)
    train_texts.pop(i)

print ("Number of training sentences :",len(train_texts)) # 4952
print ("Number of unique intents :",len(set(train_labels))) # 17 类

for i in zip(train_texts[:5],train_labels[:5]):
    print(i)
'''
('i want to fly from boston at 838 am and arrive in denver at 1110 in the morning', 'atis_flight')
('what flights are available from pittsburgh to baltimore on thursday morning', 'atis_flight')
('what is the arrival time in san francisco for the 755 am flight leaving washington', 'atis_flight_time')
('cheapest airfare from tacoma to orlando', 'atis_airfare')
('round trip fares from pittsburgh to philadelphia under 1000 dollars', 'atis_airfare')
'''

sents,labels,intents = fetch_data('Data/data2/atis.test.w-intent.iob')
test_sentences = [" ".join(i) for i in sents]
test_texts = test_sentences
test_labels = intents.tolist()

new_labels = set(test_labels) - set(train_labels)
vals = []
for i in range(len(test_labels)):
    if "#" in test_labels[i]:
        vals.append(i)
    elif test_labels[i] in new_labels:
        print(test_labels[i])
        vals.append(i)
        
for i in vals[::-1]:
    test_labels.pop(i)
    test_texts.pop(i)

print ("Number of testing sentences :",len(test_texts)) # 876
print ("Number of unique intents :",len(set(test_labels))) # 15

for i in zip(test_texts[:5], test_labels[:5]):
    print(i)
'''
('i would like to find a flight from charlotte to las vegas that makes a stop in st. louis', 'atis_flight')
('on april first i need a ticket from tacoma to san jose departing before 7 am', 'atis_airfare')
('on april first i need a flight going from phoenix to san diego', 'atis_flight')
('i would like a flight traveling one way from phoenix to san diego on april first', 'atis_flight')
('i would like a flight from orlando to salt lake city for april first on delta airlines', 'atis_flight')
'''

MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.3

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts) # Converting text to a vector of word indexes
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) 

le = LabelEncoder()
le.fit(train_labels)
train_labels = le.transform(train_labels)
test_labels = le.transform(test_labels)

trainvalid_data = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH)
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
trainvalid_labels = to_categorical(train_labels) # 17类

test_labels = to_categorical(np.asarray(test_labels), num_classes= trainvalid_labels.shape[1]) # num_classes for match 15-->17

# split the training data into a training set and a validation set
indices = np.arange(trainvalid_data.shape[0])
np.random.shuffle(indices)
trainvalid_data = trainvalid_data[indices]
trainvalid_labels = trainvalid_labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * trainvalid_data.shape[0])
x_train = trainvalid_data[:-num_validation_samples]
y_train = trainvalid_labels[:-num_validation_samples]
x_val = trainvalid_data[-num_validation_samples:]
y_val = trainvalid_labels[-num_validation_samples:]


# prepare Glove embedding indexes
import os
BASE_DIR = 'Data'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print('Found %s word vectors in Glove embeddings.' % len(embeddings_index))

# prepare embedding matrix - rows are the words from word_index, columns are the embeddings of that word from glove.
num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# load these pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print("Preparing of embedding matrix is done")

# CNN with Glove PreTrained Embeddings
cnnmodel = Sequential()
'''
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
'''
cnnmodel.add(embedding_layer)
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPooling1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPooling1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(GlobalMaxPooling1D())
cnnmodel.add(Dense(128, activation='relu'))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Dense(trainvalid_labels.shape[1], activation='softmax'))
cnnmodel.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
cnnmodel.summary()

cnnmodel.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          validation_data=(x_val, y_val))

print(cnnmodel.summary())
loss, acc = cnnmodel.evaluate(test_data, test_labels, verbose=0)
print('Test accuracy:', acc)


# CNN with a Embedding layer which is being trained on the fly instead of using the pretrained embeddings
cnnmodel = Sequential()
cnnmodel.add(Embedding(MAX_NUM_WORDS, 128))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPooling1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPooling1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(GlobalMaxPooling1D())
cnnmodel.add(Dense(128, activation='relu'))
cnnmodel.add(Dense(len(trainvalid_labels[0]), activation='softmax'))

cnnmodel.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

cnnmodel.summary()
cnnmodel.fit(x_train, y_train,
          batch_size=128,
          epochs=1, validation_data=(x_val, y_val))
#Evaluate on test set:
score, acc = cnnmodel.evaluate(test_data, test_labels)
print('Test accuracy with CNN:', acc)

# LSTM model, using Glove pre-trained embedding layer
rnnmodel2 = Sequential()
rnnmodel2.add(embedding_layer)
rnnmodel2.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel2.add(Dense(len(trainvalid_labels[0]), activation='softmax'))
rnnmodel2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

rnnmodel2.summary()

print('Training the RNN')
rnnmodel2.fit(x_train, y_train,
          batch_size=32,
          epochs=1,
          validation_data=(x_val, y_val))
score, acc = rnnmodel2.evaluate(test_data, test_labels,
                            batch_size=32)
print('Test accuracy with RNN:', acc)



#  LSTM model, training embedding layer on the fly
rnnmodel = Sequential()
rnnmodel.add(Embedding(MAX_NUM_WORDS, 128))
rnnmodel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel.add(Dense(len(trainvalid_labels[0]), activation='softmax'))

rnnmodel.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

rnnmodel.summary()

rnnmodel.fit(x_train, y_train,
          batch_size=32,
          epochs=1,
          validation_data=(x_val, y_val))
score, acc = rnnmodel.evaluate(test_data, test_labels,
                            batch_size=32)