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
# 1. CRF SNIPS dataset slot filling
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

