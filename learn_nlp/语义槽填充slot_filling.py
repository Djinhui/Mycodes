# 《自然语言处理实战：从入门到项目实践》CH06 https://github.com/practical-nlp/practical-nlp-code
'''
口语理解(Spoken Language Understanding, SLU)作为语音识别与自然语言处理之间的一个新兴领域，其目的是为了让计算机从用户的讲话中理解他们的意图。
SLU是口语对话系统Spoken Dialog Systems的一个非常关键的环节。下图展示了口语对话系统的主要流程。

SLU主要通过如下三个子任务来理解用户的语言:
 - 领域识别(Domain Detection)
 - 用户意图检测(User Intent Determination)
 - 语义槽填充(Semantic Slot Filling)

例如，用户输入“播放周杰伦的稻香”，首先通过领域识别模块识别为"music"领域，再通过用户意图检测模块识别出用户意图为"play_music"（而不是"find_lyrics" )
最后通过槽填充对将每个词填充到对应的槽中："播放[O] / 周杰伦[B-singer] / 的[O] / 稻香[B-song]"。
从上述例子可以看出，**通常把领域识别和用户意图检测当做文本分类问题，而把槽填充当做序列标注(Sequence Tagging)问题**，也就是把连续序列中每个词赋予相应的语义类别标签。

'''


# ----------------------------------------------------1. CRF SNIPS dataset slot filling----------------------------------------------------------
import os
import json
import string
from zipfile import ZipFile
import numpy as np
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer, confusion_matrix, f1_score, classification_report
from sklearn.pipeline import Pipeline

# import tensorflow as tf
# import keras
# from tensorflow.keras.preprocessing.text import Tokenizer


train_loc = "Data/snips/train_PlayMusic_full.json"
test_loc = "Data/snips/validate_PlayMusic.json"

train_file = json.load(open(train_loc, encoding='iso-8859-2'))
test_file = json.load(open(test_loc, encoding='iso-8859-2'))

train_datafile = [i['data'] for i in train_file['PlayMusic']]
test_datafile = [i['data'] for i in test_file['PlayMusic']]
'''
train_datafile[0]:
[{'text': 'I need to hear the '},
 {'text': 'song', 'entity': 'music_item'},
 {'text': ' '},
 {'text': 'Aspro Mavro', 'entity': 'track'},
 {'text': ' from '},
 {'text': 'Bill Szymczyk', 'entity': 'artist'},
 {'text': ' on '},
 {'text': 'Youtube', 'entity': 'service'}]

'''

def convert_data(datalist):
    output = []
    for data in datalist:
        sent = []
        pos = []
        for phrase in data:
            words = phrase['text'].strip().split(' ')
            while '' in words: 
                words.remove('')
            
            if 'entity' in phrase.keys():
                label = phrase['entity']
                labels = [label +'-{}'.format(i+1) for i in range(len(words))]
            else:
                labels = ['O' for i in range(len(words))]

            sent.extend(words)
            pos.extend(labels)
        output.append((sent, pos))
    return output


train_data = convert_data(train_datafile)
test_data = convert_data(test_datafile)
'''
train_data[0]:
[['I','need','to','hear','the','song','Aspro','Mavro','from','Bill','Szymczyk','on','Youtube'],
 ['O','O','O','O','O','music_item-1','track-1','track-2','O','artist-1','artist-2','O','service-1']]
'''

GLOVE_DIR = os.path.join('BASE_DIR', 'glove.6B')
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 20000 
EMBEDDING_DIM = 100 
VALIDATION_SPLIT = 0.3

embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

def get_embeddings(word):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is None:
        embedding_vector = np.zeros(shape=(EMBEDDING_DIM,))
    return embedding_vector


# train_texts = [" ".join(i[0]) for i in train_data]
# test_texts = [" ".join(i[0]) for i in test_data]

# train_texts[0] # 'I need to hear the song Aspro Mavro from Bill Szymczyk on Youtube'

# tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# tokenizer.fit_on_texts(train_texts)
# train_sequences = tokenizer.texts_to_sequences(train_texts)
# test_sequences = tokenizer.texts_to_sequences(test_texts)

# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))



def sent2feats(sentence):
    """
    Get features for all words in the sentence
    Features:
    - word context: a window of 2 words on either side of the current word, and current word.
    - POS context: a window of 2 POS tags on either side of the current word, and current tag. 
    input: sentence as a list of tokens.
    output: list of dictionaries. each dict represents features for that word.
    """
    feats = []
    sen_tags = pos_tag(sentence)
    for i in range(0,len(sentence)):
        word = sentence[i]
        wordfeats = {}
       #word features: word, prev 2 words, next 2 words in the sentence.
        wordfeats['word'] = word
        if i == 0:
            wordfeats["prevWord"] = wordfeats["prevSecondWord"] = "<S>"
        elif i==1:
            wordfeats["prevWord"] = sentence[0]
            wordfeats["prevSecondWord"] = "</S>"
        else:
            wordfeats["prevWord"] = sentence[i-1]
            wordfeats["prevSecondWord"] = sentence[i-2]
        #next two words as features
        if i == len(sentence)-2:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = "</S>"
        elif i==len(sentence)-1:
            wordfeats["nextWord"] = "</S>"
            wordfeats["nextNextWord"] = "</S>"
        else:
            wordfeats["nextWord"] = sentence[i+1]
            wordfeats["nextNextWord"] = sentence[i+2]
        
        #POS tag features: current tag, previous and next 2 tags.
        wordfeats['tag'] = sen_tags[i][1]
        if i == 0:
            wordfeats["prevTag"] = wordfeats["prevSecondTag"] = "<S>"
        elif i == 1:
            wordfeats["prevTag"] = sen_tags[0][1]
            wordfeats["prevSecondTag"] = "</S>"
        else:
            wordfeats["prevTag"] = sen_tags[i - 1][1]

            wordfeats["prevSecondTag"] = sen_tags[i - 2][1]
            # next two words as features
        if i == len(sentence) - 2:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = "</S>"
        elif i == len(sentence) - 1:
            wordfeats["nextTag"] = "</S>"
            wordfeats["nextNextTag"] = "</S>"
        else:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
            wordfeats["nextNextTag"] = sen_tags[i + 2][1]
            
        #Adding word vectors
        vector = get_embeddings(word)
        for iv, value in enumerate(vector):
            wordfeats['v{}'.format(iv)] = value

        #That is it! You can add whatever you want!
        feats.append(wordfeats)
    return feats


def get_feats_conll(conll_data):
    feats, labels = [], []
    for sentence in conll_data:
        feats.append(sent2feats(sentence[0]))
        labels.append(sentence[1])
    return feats, labels

#source for this function: https://gist.github.com/zachguo/10296432
def print_cm(cm, labels):
    print("\n")
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        sum = 0
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            sum =  sum + int(cell)
            print(cell, end=" ")
        print(sum) #Prints the total number of instances per cat at the end.

#python-crfsuite does not have a confusion matrix function, 
#so writing it using sklearn's confusion matrix and print_cm from github
def get_confusion_matrix(y_true,y_pred,labels):
    trues,preds = [], []
    for yseq_true, yseq_pred in zip(y_true, y_pred):
        trues.extend(yseq_true)
        preds.extend(yseq_pred)
    print_cm(confusion_matrix(trues,preds,labels),labels)


def train_seq(feats, labels, devfeats, devlabels):
    """
    feats/devfeats:list of lists of dicts
    labels/devlabels:list of lists of strings
    """
    crf = CRF(algorithm='lbfgs',
              c1=0.1,
              c2=0.1,
              max_iterations=100,
              all_possible_transitions=True)
    crf.fit(feats, labels)
    labels = list(crf.classes_)

    y_pred = crf.predict(devfeats)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

    print(metrics.flat_f1_score(devlabels, y_pred,average='weighted', labels=labels))
    print(metrics.flat_classification_report(devlabels, y_pred, labels=sorted_labels, digits=3))
    #print(metrics.sequence_accuracy_score(devlabels, y_pred))
    get_confusion_matrix(devlabels, y_pred,labels=sorted_labels)



feats, labels = get_feats_conll(train_data)
'''
feats[0]:
[{'word': 'I',
  'prevWord': '<S>',
  'prevSecondWord': '<S>',
  'nextWord': 'need',
  'nextNextWord': 'to',
  'tag': 'PRP',
  'prevTag': '<S>',
  'prevSecondTag': '<S>',
  'nextTag': 'VBP',
  'nextNextTag': 'TO'},
 {'word': 'need',
  'prevWord': 'I',
  'prevSecondWord': '</S>',
  'nextWord': 'to',
  'nextNextWord': 'hear',
  'tag': 'VBP',
  'prevTag': 'PRP',
  'prevSecondTag': '</S>',
  'nextTag': 'TO',
  'nextNextTag': 'VB'},
 {'word': 'to',
  'prevWord': 'need',
  'prevSecondWord': 'I',
  'nextWord': 'hear',
  'nextNextWord': 'the',
...
  'tag': 'NNP',
  'prevTag': 'IN',
  'prevSecondTag': 'NNP',
  'nextTag': '</S>',
  'nextNextTag': '</S>'}]


labels[0]:
['O',
 'O',
 'O',
 'O',
 'O',
 'music_item-1',
 'track-1',
 'track-2',
 'O',
 'artist-1',
 'artist-2',
 'O',
 'service-1']
'''


devfeats, devlabels = get_feats_conll(test_data)
train_seq(feats, labels, devfeats, devlabels)





# ----------------------------------------------------1. BERT SNIPS dataset slot filling----------------------------------------------------------
import json
import re
import os
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertConfig, BertAdam
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from keras.api.preprocessing.sequence import pad_sequences
# from keras.api.layers import TextVectorization

train_loc = "Data/snips/train_PlayMusic_full.json"
test_loc = "Data/snips/validate_PlayMusic.json"

train_file = json.load(open(train_loc, encoding= "iso-8859-2"))
test_file = json.load(open(test_loc, encoding= "iso-8859-2"))

train_datafile = [i["data"] for i in train_file["PlayMusic"]]
test_datafile = [i["data"] for i in test_file["PlayMusic"]]

def convert_data(datalist):
    output = []
    for data in datalist:
        sent = []
        pos = []
        for phrase in data:
            words = phrase['text'].strip().split(' ')
            while '' in words: 
                words.remove('')
            
            if 'entity' in phrase.keys():
                label = phrase['entity']
                labels = [label +'-{}'.format(i+1) for i in range(len(words))]
            else:
                labels = ['O' for i in range(len(words))]

            sent.extend(words)
            pos.extend(labels)
        output.append((sent, pos))
    return output


train_data = convert_data(train_datafile)
test_data = convert_data(test_datafile)
'''
train_data[0]:
[['I','need','to','hear','the','song','Aspro','Mavro','from','Bill','Szymczyk','on','Youtube'],
 ['O','O','O','O','O','music_item-1','track-1','track-2','O','artist-1','artist-2','O','service-1']]
'''

df_train = pd.DataFrame(train_data, columns=['sentence', 'label'])
df_test = pd.DataFrame(test_data, columns=['sentence', 'label'])
df_train.head()
'''
	sentence	                                        label
0	[I, need, to, hear, the, song, Aspro, Mavro, f...	[O, O, O, O, O, music_item-1, track-1, track-2...
1	[play, Yo, Ho, from, the, new, york, pops, on,...	[O, track-1, track-2, O, artist-1, artist-2, a...
2	[Play, some, seventies, music, by, Janne, Puur...	[O, O, year-1, O, O, artist-1, artist-2, O, se...
3	[play, the, MĂşsica, Da, SĂŠrie, De, Filmes, O...	[O, O, album-1, album-2, album-3, album-4, alb...
4	[Play, Magic, Sam, from, the, thirties]         	[O, artist-1, artist-2, O, O, year-1]
'''


assert len(df_train['sentence'][10]) == len(df_train['label'][10])


sentence = list(df_train['sentence']) + list(df_test['sentence']) # list of lists of strings
label = list(df_train['label']) + list(df_test['label']) # list of lists of strings

unique_labels = []
for i in label:
    unique_labels += i
labels = unique_labels # a list
unique_labels = set(unique_labels)
list(unique_labels)[:10]
'''
['genre-4',
 'track-7',
 'album-2',
 'playlist-4',
 'artist-3',
 'artist-5',
 'service-1',
 'artist-4',
 'playlist-3',
 'track-4']
'''

sentence[:2]
'''
[['I','need','to','hear','the','song','Aspro','Mavro','from','Bill','Szymczyk','on','Youtube'],
 ['play', 'Yo', 'Ho', 'from', 'the', 'new', 'york', 'pops', 'on', 'Youtube']]
'''

label[:2]
'''
[['O','O','O','O','O','music_item-1','track-1','track-2','O','artist-1','artist-2','O','service-1'],
 ['O','track-1','track-2','O','artist-1','artist-2','artist-3','artist-4','O','service-1']]
'''

def untokenize(words):
    '''
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    '''
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()


sentences_untokenized = [untokenize(i) for i in sentence] # a list of sentences
sentences_untokenized[0]  # 'I need to hear the song Aspro Mavro from Bill Szymczyk on Youtube'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpus = torch.cuda.device_count()

MAX_LEN = 75
bs = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [['[CLS]'] + tokenizer.tokenize(sent) + ['[SEP]'] for sent in sentences_untokenized]


sentence[:2]
'''
[['I','need','to','hear','the','song','Aspro','Mavro','from','Bill','Szymczyk','on','Youtube'], # 长度13
 ['play', 'Yo', 'Ho', 'from', 'the', 'new', 'york', 'pops', 'on', 'Youtube']]
'''


sentences_untokenized[:2]
'''
['I need to hear the song Aspro Mavro from Bill Szymczyk on Youtube',
 'play Yo Ho from the new york pops on Youtube']
'''

tokenized_texts[:2]
'''
[['[CLS]', 'i', 'need', 'to', 'hear', 'the', 'song', 'as', '##pro', 'ma', '##vr', '##o', 'from', 'bill', 's', '##zy', '##mc', '##zy', '##k', 'on', 'youtube', '[SEP]'], # 长度22
 ['[CLS]', 'play', 'yo', 'ho', 'from', 'the', 'new', 'york', 'pops', 'on', 'youtube', '[SEP]']]
'''

# 这样不是和label 长度不对应了吗？？？？？？？？
label[:2]
'''
[['O','O','O','O','O','music_item-1','track-1','track-2','O','artist-1','artist-2','O','service-1'], # 长度13

 ['O','track-1','track-2','O','artist-1','artist-2','artist-3','artist-4','O','service-1']]
'''


tags_vals = list(unique_labels)
tag2idx = {t:i for i, t in enumerate(tags_vals)}


input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts], maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in label], maxlen=MAX_LEN, value=tag2idx["O"], padding="post", dtype="long", truncating="post")

input_ids[:2]
'''
array([[  101,  1045,  2342,  2000,  2963,  1996,  2299,  2004, 21572,
         5003, 19716,  2080,  2013,  3021,  1055,  9096, 12458,  9096,
         2243,  2006,  7858,   102,     0,     0,     0,     0,     0,  # 非填充长22
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0],
       [  101,  2377, 10930,  7570,  2013,  1996,  2047,  2259, 16949,
         2006,  7858,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0]])
'''

tags[:2]
'''
array([[23, 23, 23, 23, 23, 18, 10,  9, 23,  3,  6, 23, 11, 23, 23, 23, # 非填充长13
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23],
       [23, 10,  9, 23,  3,  6, 16, 27, 23, 11, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23]])
'''

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]


tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)


tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))

if torch.cuda.is_available():
    model.cuda()
else:
    model

#Before starting fine tuing we need to add the optimizer. Generally Adam is used
#weight_decay is added as regularization to the main weight matrices
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
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

#accuracy
def flat_accuracy(preds, labels):
    '''
    preds:(bs, seq_len, num_classes)
    labesl:(bs, seq_len)
    '''
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


epochs = 5
max_grad_norm = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loss_set = []
for _ in range(epochs):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
       
        #forward pass
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss.item())
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        #backward pass
        loss.backward()
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        #update the parameters
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
