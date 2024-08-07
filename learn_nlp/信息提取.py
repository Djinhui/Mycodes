# 《自然语言处理实战：从入门到项目实践》CH05 https://github.com/practical-nlp/practical-nlp-code

# 1. 关键词提取
import spacy
import textacy.ke
from textacy import *

en = textacy.load_spacy_lang("en_core_web_sm")
mytext = open('Data/nlphistory.txt').read()

#convert the text into a spacy document.
doc = textacy.make_spacy_doc(mytext, lang=en)

textacy.ke.textrank(doc, topn=5)

#Print the keywords using TextRank algorithm, as implemented in Textacy.
print("Textrank output: ", [kps for kps, weights in textacy.ke.textrank(doc, normalize="lemma", topn=5)])\
#Print the key words and phrases, using SGRank algorithm, as implemented in Textacy
print("SGRank output: ", [kps for kps, weights in textacy.ke.sgrank(doc, topn=5)])

#To address the issue of overlapping key phrases, textacy has a function: aggregage_term_variants.
#Choosing one of the grouped terms per item will give us a list of non-overlapping key phrases!
terms = set([term for term,weight in textacy.ke.sgrank(doc)])
print(textacy.ke.utils.aggregate_term_variants(terms))

# 2. 命名实体识别

# 2.1 train a ner model using CONLL dataset and `sklearn_crfsuite` library
from nltk.tag import pos_tag
from sklearn_crfsuite import CRF, metrics
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
import string
import warnings
warnings.filterwarnings("ignore")

train_path = 'Data/conlldata/train.txt'
test_path = 'Data/conlldata/test.txt'
"""
文本格式
EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
.	O

...
"""

def load_data_conll(file_path):
    """
    Load the training/testing data. 
    input: conll format data, but with only 2 tab separated colums - words and NEtags.
    output: A list where each item is 2 lists.  sentence as a list of tokens, NER tags as a list for each token.
    """
    myoutput, words, tags = [], [],[]
    fh = open(file_path)    
    for line in fh:
        line = line.strip()
        if '\t' not in line:
            # sentence ended
            myoutput.append([words,tags])
            words, tags = [],[]
        else:
            word, tag = line.split('\t')
            words.append(word)
            tags.append(tag)
    fh.close()
    return myoutput


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
    sen_tags = pos_tag(sentence) #This format is specific to this POS tagger.
    for i in range(len(sentence)):
        word = sentence[i]
        wordfeats = {}
        # word features: word, prev 2 words, next 2 words in the sentence.
        wordfeats['word'] = word
        if i == 0:
            wordfeats['prevWord'] = wordfeats['prevSencondWord'] = '<S>'
        elif i == 1:
            wordfeats['prevWord'] = sentence[0]
            wordfeats['prevSencondWord'] = '</S>'
        else:
            wordfeats['prevWord'] = sentence[i-1]
            wordfeats['prevSencondWord'] = sentence[i-2]

        if i == len(sentence) - 2:
            wordfeats['nextWord'] = sentence[i+1]
            wordfeats['nextNextWord'] = '</S>'
        elif i == len(sentence) - 1:
            wordfeats['nextWord'] = '</s>'
            wordfeats['nextNextWord'] = '</S>'
        else:
            wordfeats['nextWord'] = sentence[i+1]
            wordfeats['nextNextWord'] = sentence[i+2]

        # POS features: current tag, prev and  next 2 tags
        wordfeats['tag'] = sen_tags[i][1]
        if i == 0:
            wordfeats['prevTag'] = wordfeats['prevSencondTag'] = '<S>'
        elif i == 1:
            wordfeats['prevTag'] = sen_tags[0][1]
            wordfeats['prevSencondTag'] = '</S>'
        else:
            wordfeats['prevTag'] = sen_tags[i-1][1]
            wordfeats['prevSencondTag'] = sen_tags[i-2][1]

        if i == len(sentence) - 2:
            wordfeats['nextTag'] = sen_tags[i+1][1]
            wordfeats['nextNextTag'] = '</S>'
        elif i == len(sentence) - 1:
            wordfeats['nextTag'] = '</s>'
            wordfeats['nextNextTag'] = '</S>'
        else:
            wordfeats['nextTag'] = sen_tags[i+1][1]
            wordfeats['nextNextTag'] = sen_tags[i+2][1]

        feats.append(wordfeats)
    return feats


def get_feats_conll(conll_data):
    feats, labels = [], []
    for sentence in conll_data:
        feats.append(sent2feats(sentence[0]))
        labels.append(sentence[1])
    return feats, labels


def train_seq(X_train,Y_train,X_dev,Y_dev):
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)#, all_possible_states=True)
    crf.fit(X_train, Y_train)
    labels = list(crf.classes_)

    y_pred = crf.predict(X_dev)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))    
    print(metrics.flat_f1_score(Y_dev, y_pred,average='weighted', labels=labels))
    print(metrics.flat_classification_report(Y_dev, y_pred, labels=sorted_labels, digits=3))
    #print(metrics.sequence_accuracy_score(Y_dev, y_pred))
    get_confusion_matrix(Y_dev, y_pred,labels=sorted_labels)
    return y_pred


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

conll_train = load_data_conll(train_path)
conll_dev = load_data_conll(test_path)

print("Training a Sequence classification model with CRF")
feats, labels = get_feats_conll(conll_train)
devfeats, devlabels = get_feats_conll(conll_dev)

train_seq(feats,labels,devfeats,devlabels)
print('Done with model')

"""
sklearn_crfsuite 结果
Training a Sequence classification model with CRF
0.9255103670420659
              precision    recall  f1-score   support

           O      0.973     0.981     0.977     38323
       B-LOC      0.694     0.765     0.728      1668
       I-LOC      0.738     0.482     0.584       257
      B-MISC      0.648     0.309     0.419       702
      I-MISC      0.626     0.505     0.559       216
       B-ORG      0.670     0.561     0.611      1661
       I-ORG      0.551     0.704     0.618       835
       B-PER      0.773     0.766     0.769      1617
       I-PER      0.819     0.886     0.851      1156

    accuracy                          0.928     46435
   macro avg      0.721     0.662     0.679     46435
weighted avg      0.926     0.928     0.926     46435



                O  B-LOC  I-LOC B-MISC I-MISC  B-ORG  I-ORG  B-PER  I-PER 
         O  37579    118      3     22     32    193    224     88     64 38323
     B-LOC    143   1276      1     36      1     95     14     98      4 1668
     I-LOC     32      6    124      1      5      0     52      0     37 257
    B-MISC    344     48      1    217      2     56     13     19      2 702
...
     I-ORG     76     15     18      2     15     21    588      8     92 835
     B-PER     86    138      1      5      3     90     44   1238     12 1617
     I-PER     26      1     16      0      4      2     83      0   1024 1156
Done with sequence model
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

"""


# 2.2 Problems of NER illustration through Spacy.
import spacy
nlp = spacy.load("en_core_web_lg") # !python -m spacy download en_core_web_lg
mytext = """SAN FRANCISCO — Shortly after Apple used a new tax law last year to bring back most of the $252 billion it had held abroad, the company said it would buy back $100 billion of its stock.

On Tuesday, Apple announced its plans for another major chunk of the money: It will buy back a further $75 billion in stock.

“Our first priority is always looking after the business and making sure we continue to grow and invest,” Luca Maestri, Apple’s finance chief, said in an interview. “If there is excess cash, then obviously we want to return it to investors.”

Apple’s record buybacks should be welcome news to shareholders, as the stock price is likely to climb. But the buybacks could also expose the company to more criticism that the tax cuts it received have mostly benefited investors and executives.
"""
doc = nlp(mytext)
for ent in doc.ents:
    print(ent.text, "\t", ent.label_)

count=0 #We see 6 sentences as humans in this text. How many does Spacy see? 
for sent in doc.sents:
    print(sent.text)
    print("***End of sent****")
    count = count+1
print("Total sentences: ", count) # 11

# 2.3 BERT-CONLL-NER
import string
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score, classification_report
from nltk.tag import pos_tag

import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

"""
Load the training/testing data. 
input: conll format data, but with only 2 tab separated colums - words and NEtags.
output: A list where each item is 2 lists.  sentence as a list of tokens, NER tags as a list for each token.
"""
#functions for preparing the data in the *.txt files
def load__data_conll(file_path):
    myoutput,words,tags = [],[],[]
    fh = open(file_path)
    for line in fh:
        line = line.strip()
        if "\t" not in line:
            #Sentence ended.
            myoutput.append([words,tags])
            words,tags = [],[]
        else:
            word, tag = line.split("\t")
            words.append(word)
            tags.append(tag)
    fh.close()
    return myoutput

"""
Get features for all words in the sentence
Features:
- word context: a window of 2 words on either side of the current word, and current word.
- POS context: a window of 2 POS tags on either side of the current word, and current tag. 
input: sentence as a list of tokens.
output: list of dictionaries. each dict represents features for that word.
"""
def sent2feats(sentence):
    feats = []
    sen_tags = pos_tag(sentence) #This format is specific to this POS tagger!
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
        #That is it! You can add whatever you want!
        feats.append(wordfeats)
    return feats

train_path = 'Data/conlldata/train.txt'
test_path = 'Data/conlldata/test.txt'
'''
conll2003/en/train.txt

EU NNP B-NP B-ORG
rejects VBZ B-VP O
German JJ B-NP B-MISC
call NN I-NP O
to TO B-VP O
boycott VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
. . O O


conlldata/train.txt

EU	B-ORG
rejects	O
German	B-MISC
call	O
to	O
boycott	O
British	B-MISC
lamb	O
.	O
'''

conll_train = load__data_conll(train_path)
conll_test = load__data_conll(test_path)

df_train = pd.read_csv(train_path, sep='\t', header=None, engine='python', encoding='utf-8')
df_test = pd.read_csv(test_path, sep='\t', header=None, engine='python', encoding='utf-8')
df = pd.merge(df_train, df_test)
label = list(df[1].values)

import re
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

#lets convert them to dataframs for easier handling
df_train = pd.DataFrame(conll_train,columns=["sentence","labels"])
df_test = pd.DataFrame(conll_test,columns=["sentence","labels"])

#getting all the sentences and labels present in both test and train
sentences = list(df_train['sentence'])+list(df_test['sentence'])
print("No of sentences:",len(sentences)) # 17494
labels = list(df_train['labels'])+list(df_test['labels']) 
print("No of labels:",len(labels)) # 17494

sentences = [untokenize(sent) for sent in sentences]
sentences[0] # 'EU rejects German call to boycott British lamb.'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
MAX_LENG = 75
bs = 32

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = list(map(lambda x: ['[CLS]'] + tokenizer.tokenize(x) + ['[SEP]'] , sentences))
print(tokenized_texts[0]) # ['[CLS]', 'eu', 'rejects', 'german', 'call', 'to', 'boycott', 'british', 'lamb', '.', '[SEP]']
print(labels[0]) # ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']

tags_vals = list(set(label))
tag2idx = {t:i for i, t in enumerate(tags_vals)}

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],maxlen=MAX_LENG, dtype="long", truncating="post", padding="post")
tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels], maxlen=MAX_LENG, value=tag2idx["O"], padding="post", dtype="long", truncating="post")

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, 
                                                            random_state=2020, test_size=0.2)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2020, test_size=0.2)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
print("Train Data Loaders Ready")
valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
print("Test Data Loaders Ready")

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx))#loading pre trained bert
print("BERT model ready to use")


if torch.cuda.is_available():    
    print("Passing Model parameters in GPU")
    print(model.cuda()) 
else: 
    print(model)

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

epochs = 4
max_grad_norm = 1.0
train_loss_set = []
for _ in trange(epochs, desc="Epoch"):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
       
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        train_loss_set.append(loss)
        loss.backward()
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        torch.nn.utils.clip_grad(model.parameters(), max_grad_norm)
        optimizer.step()
        model.zero_grad()
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.ylim(0,0.25)
plt.plot(train_loss_set)
plt.show()


#Evaluate the model
model.eval()
predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in valid_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    with torch.no_grad():
        tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits.detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]
print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))


# 3. 实体链接
# extract Named Entity linking information using [Azure Text Analytics API]
import requests
import pprint

my_api_key = 'xxxx' #replace this with your api key

def print_entities(text):
    url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.1/entities"
    documents = {'documents':[{'id':'1', 'language':'en', 'text':text}]}
    headers = {'Ocp-Apim-Subscription-Key': my_api_key}
    response = requests.post(url, headers=headers, json=documents)
    entities = response.json()
    return entities

mytext = open("nytarticle.txt").read() #This file is in the same folder. 
entities = print_entities(mytext)
for document in entities["documents"]:
        pprint.pprint(document["entities"])
#This above code will print you a whole lot of stuff you may or may not use later.
'''
[{'bingId': '37181124-e096-403d-a455-576a61b83525',
  'matches': [{'entityTypeScore': 0.7327625155448914,
               'length': 13,
               'offset': 0,
               'text': 'SAN FRANCISCO',
               'wikipediaScore': 0.12144925273060747}],
  'name': 'San Francisco',
  'type': 'Location',
  'wikipediaId': 'San Francisco',
  'wikipediaLanguage': 'en',
  'wikipediaUrl': 'https://en.wikipedia.org/wiki/San_Francisco'},
 {'bingId': '0906eb9b-9868-63ec-4602-7c378ae70164',
  'matches': [{'entityTypeScore': 0.8,
               'length': 5,
               'offset': 16,
               'text': 'After',
               'wikipediaScore': 0.00812327874600638}],
  'name': '(after)',
  'type': 'Other',
  'wikipediaId': '(after)',
  'wikipediaLanguage': 'en',
  'wikipediaUrl': 'https://en.wikipedia.org/wiki/(after)'},
  ......

'''
#Let us clean up a little bit, and not print the whole lot of messy stuff it gives us?
for document in entities['documents']:
    print("Entities in this document: ")
    for entity in document['entities']:
        if entity['type'] in ["Person", "Location", "Organization"]:
            print(entity['name'], "\t", entity['type'])
            if 'wikipediaUrl' in entity.keys():
                print(entity['wikipediaUrl'])

'''
Entities in this document: 
San Francisco 	 Location
https://en.wikipedia.org/wiki/San_Francisco
Facebook 	 Organization
https://en.wikipedia.org/wiki/Facebook
Alex Jones 	 Person
https://en.wikipedia.org/wiki/Alex_Jones
InfoWars 	 Organization
https://en.wikipedia.org/wiki/InfoWars
Louis Farrakhan 	 Person
https://en.wikipedia.org/wiki/Louis_Farrakhan
Silicon Valley 	 Location
https://en.wikipedia.org/wiki/Silicon_Valley
Instagram 	 Organization
https://en.wikipedia.org/wiki/Instagram
us 	 Location
'''

# 4. 关系提取
# Relation Extraction with IBM Watson. 
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, RelationsOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey='XXXXX',
    url='https://gateway-wdc.watsonplatform.net/natural-language-understanding/api'
)

response = natural_language_understanding.analyze(
    text='Leonardo DiCaprio won Best Actor in a Leading Role for his performance.',
    features=Features(relations=RelationsOptions())).get_result()

print(json.dumps(response, indent=2))

mytext = "Satya Narayana Nadella currently serves as the Chief Executive Officer (CEO) of Microsoft."
response = natural_language_understanding.analyze(
    text=mytext,
    features=Features(relations=RelationsOptions())).get_result()
result = json.dumps(response)
print(result)
'''
'{"usage": {"text_units": 1, "text_characters": 90, "features": 1}, 
"relations": [{"type": "employedBy", "sentence": "Satya Narayana Nadella currently serves as the Chief Executive Officer (CEO) of Microsoft.", "score": 0.48706, "arguments": [{"text": "CEO", "location": [72, 75], "entities": [{"type": "Person", "text": "Satya Narayana Nadella"}]}, {"text": "Microsoft", "location": [80, 89], "entities": [{"type": "Organization", "text": "Microsoft", "disambiguation": {"subtype": ["Commercial"]}}]}]}], "language": "en"}'

'''



# 5. 其他信息提取
