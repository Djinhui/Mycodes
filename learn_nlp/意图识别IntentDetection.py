# 《自然语言处理实战：从入门到项目实践》CH06 https://github.com/practical-nlp/practical-nlp-code








# 1. 意图检测、意图识别-----本质上是有监督多分类任务
# ---------------------------------------------------------1.1 CNN and RNN for the task of intent detection on the ATIS dataset-------------------------------------------------
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



# 1.2 ----------------------------------------意图分类 For the ATIS Dataset using BERT--------------------------------------------------------------------------
import os
import numpy as np
import pickle
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch.optim as optim
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertAdam, BertForSequenceClassification
import io
import pandas as pd
import matplotlib.pyplot as plt
from seqeval.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

sents,labels,intents = get_data2('Data/data2/atis.train.w-intent.iob')

train_sentences = [" ".join(i) for i in sents]
train_texts = train_sentences
train_labels= intents.tolist()

vals = []
for i in range(len(train_labels)):
    if "#" in train_labels[i]:
        vals.append(i)
        
for i in vals[::-1]:
    train_labels.pop(i)
    train_texts.pop(i)

sents,labels,intents = get_data('Data/data2/atis.test.w-intent.iob')

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

from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame(data =zip(train_labels,train_texts),columns=['Labels',"Text"])

lb_make = LabelEncoder()
df["Labels"] = lb_make.fit_transform(df["Labels"])

train_texts = list(df['Text'])
train_labels = list(df['Labels'])

query_data_train = list(train_texts)
sentences = ["[CLS] " + query + " [SEP]" for query in query_data_train]
print(sentences[0]) # [CLS] i want to fly from boston at 838 am and arrive in denver at 1110 in the morning [SEP]

#BERT only takes a token size of 512.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0]) # ['[CLS]', 'i', 'want', 'to', 'fly', 'from', 'boston', 'at', '83', '##8', 'am', 'and', 'arrive', 'in', 'denver', 'at', '111', '##0', 'in', 'the', 'morning', '[SEP]']

# Set the maximum sequence length. 
MAX_LEN = 128
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)

intent_data_label_train = train_labels #renaming 


train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, 
                                                                                    intent_data_label_train, 
                                                                                    random_state=2020, 
                                                                                    test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks,
                                                       input_ids,
                                                       random_state=2020, 
                                                       test_size=0.1)

train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=17)
if torch.cuda.is_available():
    model.cuda()
else:
    model

# BERT fine-tuning parameters
FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.01},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
      'weight_decay_rate': 0.0}
  ]
else:
    param_optimizer = list(model.classifier.named_parameters()) 
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = BertAdam(optimizer_grouped_parameters, lr=3e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Store our loss and accuracy for plotting
train_loss_set = []
# Number of training epochs 
epochs = 4

# BERT training loop
for _ in range(epochs):
    ###### TRAINING ######
  
    # Set our model to training mode
    model.train()  
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())    
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
       
    ###### VALIDATION ######

    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# plot training performance
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()







import pickle
#preprocessing functions
def get_data(filename):
    df = pd.read_csv(filename,delim_whitespace=True,names=['word','label'])
    beg_indices = list(df[df['word'] == 'BOS'].index)+[df.shape[0]]
    sents,labels,intents = [],[],[]
    for i in range(len(beg_indices[:-1])):
        sents.append(df[beg_indices[i]+1:beg_indices[i+1]-1]['word'].values)
        labels.append(df[beg_indices[i]+1:beg_indices[i+1]-1]['label'].values)
        intents.append(df.loc[beg_indices[i+1]-1]['label'])    
    return np.array(sents),np.array(labels),np.array(intents)

def get_data2(filename):
    with open(filename) as f:
        contents = f.read()
    sents,labels,intents = [],[],[]
    for line in contents.strip().split('\n'):
        words,labs = [i.split(' ') for i in line.split('\t')]
        sents.append(words[1:-1])
        labels.append(labs[1:-1])
        intents.append(labs[-1])
    return np.array(sents),np.array(labels),np.array(intents)

sents,labels,intents = get_data2('Data/data2/atis.train.w-intent.iob')
train_sentences = [" ".join(i) for i in sents]
train_texts = train_sentences
train_labels= intents.tolist()

vals = []
for i in range(len(train_labels)):
    if "#" in train_labels[i]:
        vals.append(i)
        
for i in vals[::-1]:
    train_labels.pop(i)
    train_texts.pop(i)


# load Pickle file 
def load_ds(fname, verbose=True):
    with open(fname, 'rb') as stream:
        ds,dicts = pickle.load(stream)
    if verbose:
        print('Done  loading: ', fname)
        print('      samples: {:4d}'.format(len(ds['query'])))
        print('   vocab_size: {:4d}'.format(len(dicts['token_ids'])))
        print('   slot count: {:4d}'.format(len(dicts['slot_ids'])))
        print(' intent count: {:4d}'.format(len(dicts['intent_ids'])))
    return ds,dicts
  
# convert Pickle file to arrays
def load_atis(filename, add_start_end_token=False, verbose=True):
    train_ds, dicts = load_ds(os.path.join(DATA_DIR,filename), verbose)
    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(train_ds.get, ['query', 'slot_labels', 'intent_labels'])

    if add_start_end_token:
        i2s[178] = 'BOS'
        i2s[179] = 'EOS'
        s2i['BOS'] = 178
        s2i['EOS'] = 179

    input_tensor = []
    target_tensor = []
    query_data = []
    intent_data = []
    slot_data = []
    to_show = np.random.randint(0, len(query)-1, 5)
    for i in range(len(query)):
        input_tensor.append(query[i])
        slot_text = []
        slot_vector = []
        for j in range(len(query[i])):
            slot_text.append(i2s[slots[i][j]])
            slot_vector.append(slots[i][j])
        if add_start_end_token:
            slot_text[0] = 'BOS'
            slot_vector[0] = 178
            slot_text[-1] = 'EOS'
            slot_vector[-1]= 179
        target_tensor.append(slot_vector)
        q = ' '.join(map(i2t.get, query[i]))
        query_data.append(q.replace('BOS', '').replace('EOS',''))
        intent_data.append(i2in[intent[i][0]])
        slot = ' '.join(slot_text)
        slot_data.append(slot[1:-1])
        if i in to_show and verbose:
            print('Query text:', q)
            print('Query vector: ', query[i])
            print('Intent label: ', i2in[intent[i][0]])
            print('Slot text: ', slot)
            print('Slot vector: ', slot_vector)
            print('*'*74)
   
    query_data = np.array(query_data)
    intent_data = np.array(intent_data)
    slot_data = np.array(slot_data)
    intent_data_label = np.array(intent).flatten()
    
    return t2i, s2i, in2i, i2t, i2s, i2in, input_tensor, target_tensor, query_data, intent_data, intent_data_label, slot_data


t2i_train, s2i_train, in2i_train, i2t_train, i2s_train, i2in_train, \
    input_tensor_train, target_tensor_train, \
    query_data_train, intent_data_train, intent_data_label_train, slot_data_train = load_atis('Data/data/atis.train.pkl')

t2i_test, s2i_test, in2i_test, i2t_test, i2s_test, i2in_test, \
    input_tensor_test, target_tensor_test, \
    query_data_test, intent_data_test, intent_data_label_test, slot_data_test = load_atis('Data/data/atis.test.pkl')

'''
Query text: BOS show me fares from seattle to minneapolis EOS
Query vector:  [178 770 581 415 444 752 851 597 179]
Intent label:  airfare
Slot text:  O O O O O B-fromloc.city_name O B-toloc.city_name O
Slot vector:  [128, 128, 128, 128, 128, 48, 128, 78, 128]
**************************************************************************
Query text: BOS i would like to find a flight from pittsburgh to boston on wednesday and i have to be in boston by one so i 'd like a flight out of here no later than 11 am EOS
Query vector:  [178 479 932 545 851 423 180 428 444 682 851 266 654 908 215 479 463 851
 250 482 266 277 656 779 479   0 545 180 428 669 646 468 627 531 823  24
 210 179]
Intent label:  flight
Slot text:  O O O O O O O O O B-fromloc.city_name O B-toloc.city_name O B-depart_date.day_name O O O O O O B-toloc.city_name B-arrive_time.time_relative B-arrive_time.time O O O O O O O O O B-depart_time.time_relative I-depart_time.time_relative I-depart_time.time_relative O O O
Slot vector:  [128, 128, 128, 128, 128, 128, 128, 128, 128, 48, 128, 78, 128, 26, 128, 128, 128, 128, 128, 128, 78, 15, 14, 128, 128, 128, 128, 128, 128, 128, 128, 128, 36, 101, 101, 128, 128, 128]
**************************************************************************
Query text: BOS show me the cheapest flight from pittsburgh to atlanta on wednesday which leaves before noon and serves breakfast EOS
Query vector:  [178 770 581 827 296 428 444 682 851 242 654 908 920 538 253 631 215 758
 269 179]
Intent label:  flight
Slot text:  O O O O B-cost_relative O O B-fromloc.city_name O B-toloc.city_name O B-depart_date.day_name O O B-depart_time.time_relative B-depart_time.time O O B-meal_description O
Slot vector:  [128, 128, 128, 128, 21, 128, 128, 48, 128, 78, 128, 26, 128, 128, 36, 35, 128, 128, 53, 128]
...
Intent label:  flight
Slot text:  O O O O O B-round_trip I-round_trip O O B-fromloc.city_name I-fromloc.city_name O B-toloc.city_name O B-depart_date.day_name B-depart_date.month_name B-depart_date.day_number I-depart_date.day_number O O B-arrive_time.time I-arrive_time.time O
Slot vector:  [128, 128, 128, 128, 128, 66, 119, 128, 128, 48, 110, 128, 78, 128, 26, 28, 27, 95, 128, 128, 14, 89, 128]
**************************************************************************

'''

query_data_train[:3]
'''
array([' i want to fly from boston at 838 am and arrive in denver at 1110 in the morning ',
       ' what flights are available from pittsburgh to baltimore on thursday morning ',
       ' what is the arrival time in san francisco for the 755 am flight leaving washington '])
'''

#function to find the length of a tensor
def max_length(tensor):
    return max(len(t) for t in tensor)

# Helper function to pad the query tensor and slot (target) tensor to the same length. 
# Also creates a tensor for teacher forcing.
def create_tensors(input_tensor, target_tensor, nb_sample=9999999, max_len=0):
    len_input, len_target  = max_length(input_tensor), max_length(target_tensor)
    len_input = max(len_input,max_len)
    len_target = max(len_target,max_len)
    

    # Padding the input and output tensor to the maximum length
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, 
                                                                 maxlen=len_input,
                                                                 padding='post')

    teacher_data = tf.keras.preprocessing.sequence.pad_sequences(target_tensor, 
                                                                  maxlen=len_target , 
                                                                  padding='post')
    
    target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
    target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")
    target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))
    
    nb = len(input_data)
    p = np.random.permutation(nb)
    input_data = input_data[p]
    teacher_data = teacher_data[p]
    target_data = target_data[p]

    return input_data[:min(nb_sample, nb)], teacher_data[:min(nb_sample, nb)], target_data[:min(nb_sample, nb)], len_input, len_target 
           
input_data_train, teacher_data_train, target_data_train, \
                  len_input_train, len_target_train  = create_tensors(input_tensor_train, target_tensor_train)
input_data_test, teacher_data_test, target_data_test, \
                 len_input_test, len_target_test  = create_tensors(input_tensor_test, target_tensor_test, max_len=len_input_train)

sentences = ["[CLS] " + query + " [SEP]" for query in query_data_train]
print(sentences[0])

# Tokenize with BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])

# Set the maximum sequence length. 
MAX_LEN = 128
# Pad our input tokens
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks
attention_masks = []
# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
  

#Reducing this into a binary classifcation problem
# intent_data_label_train[intent_data_label_train==14] = -1
intent_data_label_train[intent_data_label_train!=-1] = 0
intent_data_label_train[intent_data_label_train==-1] = 1

# Use train_test_split to split our data into train and validation sets for training
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, intent_data_label_train, 
                                                            random_state=2018, test_size=0.1)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
                                             
# Convert all of our data into torch tensors, the required datatype for our model
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Select a batch size for training. 
batch_size = 32

# Create an iterator of our data with torch DataLoader 
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

if torch.cuda.is_available():
    model.cuda()
else :
    model


# BERT fine-tuning parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=2e-5,
                     warmup=.1)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
  
# Store our loss and accuracy for plotting
train_loss_set = []
# Number of training epochs 
epochs = 4

# BERT training loop
for _ in range(epochs, desc="Epoch"):  
  
  ## TRAINING
  
  # Set our model to training mode
    model.train()  
    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
  # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())    
        # Backward pass
        loss.backward()
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
       
  ## VALIDATION

    # Put model in evaluation mode
    model.eval()
    # Tracking variables 
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    # Evaluate data for one epoch
    for batch in validation_dataloader:
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        with torch.no_grad():
          # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)    
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# plot training performance
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)
plt.show()