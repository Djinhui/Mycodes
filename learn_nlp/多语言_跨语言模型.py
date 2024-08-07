# 《精通transformer》 CH9

# 1. mBERT:多语言填空

from transformers import pipeline

unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')

sentences = [ 
"Transformers changed the [MASK] language processing", 
"Transformerlar [MASK] dil işlemeyi değiştirdiler", 
"ترنسفرمرها پردازش زبان [MASK] را تغییر دادند" 
] 
for sentence in sentences: 
    print(sentence) 
    print(unmasker(sentence)[0]["sequence"]) 
    print("="*50)


# 2. XLM：预训练目标：因果语言建模、带掩码机制的语言建模、翻译语言建模(同一个句子的不同语言表示，构成句子对)
unmasker = pipeline('fill-mask', model='xlm-roberta-base')
sentences = [ 
"Transformers changed the <mask> language processing", 
"Transformerlar <mask> dil işlemeyi değiştirdiler", 
"ترنسفرمرها پردازش زبان <mask> را تغییر دادند" 
]
for sentence in sentences: 
    print(sentence) 
    print(unmasker(sentence)[0]["sequence"]) 
    print("="*50) 

print(unmasker("Transformers changed the natural language processing. </s> Transformerlar <mask> dil işlemeyi değiştirdiler.")[0]["sequence"]) 
print(unmasker("Earth is a great place to live in. </s> زمین جای خوبی برای <mask> کردن است.")[0]["sequence"]) 

# 3. 跨语言文本相似性
# XLM-R
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('sts-xlm-r-multilingual')
# model = SentenceTransformer('LaBSE')

# 阿塞拜疆语
azeri_sentences = ['Pişik çöldə oturur', 
              'Bir adam gitara çalır', 
              'Mən makaron sevirəm', 
              'Yeni film möhtəşəmdir', 
              'Pişik bağda oynayır', 
              'Bir qadın televizora baxır', 
              'Yeni film çox möhtəşəmdir', 
              'Pizzanı sevirsən?'] 
# 英语
english_sentences = ['The cat sits outside', 
             'A man is playing guitar', 
             'I love pasta', 
             'The new movie is awesome',
             'The cat plays in the garden', 
             'A woman watches TV', 
             'The new movie is so great', 
             'Do you like pizza?'] 

azeri_representation = model.encode(azeri_sentences) 
english_representation = model.encode(english_sentences) 

# 在另一种语言的表示中搜索第一种语言在语义上相似的句子
results = []
for azeri_sentence, query in zip(azeri_sentences, azeri_representation):
    id_, score = util.semantic_search(query, english_representation)[0][0].values()
    results.append({
        "azeri": azeri_sentence,
        "english": english_sentences[id_],
        "score": round(score, 4)
    })

import pandas as pd 
pd.DataFrame(results) 

# 4. 可视化跨语言文本相似性
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import umap
import pylab


data = load_dataset("xtreme","tatoeba.rus", split="validation")
pd.DataFrame(data)[["source_sentence","target_sentence"]] # 一个句子是另一个句子的翻译

model = SentenceTransformer('stsb-xlm-r-multilingual')

K = 30
emb = model.encode(data["source_sentence"][:K]  + data["target_sentence"][:K])
len(emb), len(emb[0]) #（60， 768）

X= umap.UMAP(n_components=2, random_state=42).fit_transform(emb)
idx= np.arange(len(emb))

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_facecolor('whitesmoke')
cm = pylab.get_cmap("prism")
colors = list(cm(1.0*i/K) for i in range(K))

for i in idx:
    if i<K:
        ax.annotate("RUS-"+str(i), (X[i,0], X[i,1]), c=colors[i])
        ax.plot((X[i,0], X[i+K,0]), (X[i,1], X[i+K,1]), "k:" )
    else:
        ax.annotate("EN-"+str(i%K), (X[i,0], X[i,1]), c=colors[i%K])

# 计算所有句子对的余弦相似性
source_emb=model.encode(data["source_sentence"])
target_emb=model.encode(data["target_sentence"])

from scipy import spatial
from matplotlib import pyplot
sims = [ 1 - spatial.distance.cosine(s,t) for s,t in zip(source_emb, target_emb)]
pyplot.hist(sims, bins=100, range=(0.8,1))
pyplot.show()


# 5. 跨语言分类
# 如何使用英语训练用于文本分类的跨语言模型，并使用其他语言进行测试
from datasets import load_dataset 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
import numpy as np
import tensorflow as tf


imdb = load_dataset("imdb")
imdb = imdb.shuffle()
imdb_x = [x for x in imdb['train'][:1000]['text']] 
labels = [x for x in imdb['train'][:1000]['label']] 

pd.DataFrame(imdb_x, 
             columns=["text"]).to_excel( 
                                 "imdb.xlsx", 
                                  index=None) 

# 将imdb.xlsx上传到Google文档翻译器，获得该数据集的高棉语翻译
imdb_khmer = list(pd.read_excel("KHMER.xlsx").text) # 高棉语， 小语种

train_x, test_x, train_y, test_y, khmer_train, khmer_test = train_test_split(imdb_x, labels, imdb_khmer, test_size=0.2)

model = SentenceTransformer('sts-xlm-r-multilingual')

encoded_train = model.encode(train_x)
encoded_test = model.encode(test_x)
encoded_khmer_train = model.encode(khmer_train)
encoded_khmer_test = model.encode(khmer_test)

train_y = np.array(train_y) 
test_y = np.array(test_y) 

input_ = tf.keras.layers.Input(shape=(768,))
classification = tf.keras.layers.Dense(1, activation='sigmoid')(input_)
classification_model = tf.keras.Model(input_, classification)
classification_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 在英语上训练
classification_model.fit(encoded_train, train_y, epochs=10, validation_data=(encoded_test, test_y))

# 在高棉语上评估
classification_model.evaluate(encoded_khmer_test, test_y)


# 6. 跨语言零样本学习
from torch.nn.functional import softmax 
from transformers import MT5ForConditionalGeneration, MT5Tokenizer 

model_name = "alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli" 
tokenizer = MT5Tokenizer.from_pretrained(model_name) 
model = MT5ForConditionalGeneration.from_pretrained(model_name)

sequence_to_classify = "Wen werden Sie bei der nächsten Wahl wählen? " 
candidate_labels = ["spor", "ekonomi", "politika"] 
hypothesis_template = "Dieses Beispiel ist {}."

ENTAILS_LABEL = "▁0" 
NEUTRAL_LABEL = "▁1" 
CONTRADICTS_LABEL = "▁2" 
label_inds = tokenizer.convert_tokens_to_ids([ 
                           ENTAILS_LABEL, 
                           NEUTRAL_LABEL, 
                           CONTRADICTS_LABEL])

def process_nli(premise, hypothesis): 
    return f'xnli: premise: {premise} hypothesis: {hypothesis}' 

pairs = [(sequence_to_classify, hypothesis_template.format(label)) for label in candidate_labels] 
seqs = [process_nli(premise=premise, 
                    hypothesis=hypothesis) 
                    for premise, hypothesis in pairs] 

print(seqs) 
'''
['xnli: premise: Wen werden Sie bei der nächsten Wahl wählen?  hypothesis: Dieses Beispiel ist spor.', 
'xnli: premise: Wen werden Sie bei der nächsten Wahl wählen?  hypothesis: Dieses Beispiel ist ekonomi.', 
'xnli: premise: Wen werden Sie bei der nächsten Wahl wählen?  hypothesis: Dieses Beispiel ist politika.']
'''

inputs = tokenizer.batch_encode_plus(seqs, return_tensors="pt", padding=True) 
out = model.generate(**inputs, output_scores=True, return_dict_in_generate=True, num_beams=1) 
scores = out.scores[0] 
scores = scores[:, label_inds] 
print(scores)
'''
tensor([[-0.9851,  2.2550, -0.0783],
        [-5.1690, -0.7202, -2.5855],
        [ 2.7442,  3.6727,  0.7169]])
'''

entailment_ind = 0 
contradiction_ind = 2 
entail_vs_contra_scores = scores[:, [entailment_ind, contradiction_ind]] 
entail_vs_contra_probas = softmax(entail_vs_contra_scores, dim=1) 
print(entail_vs_contra_probas)
'''
tensor([[0.2877, 0.7123],
        [0.0702, 0.9298],
        [0.8836, 0.1164]])
'''

entail_scores = scores[:, entailment_ind] 
entail_probas = softmax(entail_scores, dim=0) 

print(dict(zip(candidate_labels, entail_probas.tolist()))) 
# {'spor': 0.023438096046447754, 'ekonomi': 0.0003571564157027751, 'politika': 0.9762046933174133}


# 7. 微调多语言模型
import torch
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import pandas as pd
data = pd.read_csv("TTC4900.csv")
data = data.sample(frac=1.0, random_state=42)
labels = ["teknoloji","ekonomi","saglik","siyaset","kultur","spor","dunya"]
NUM_LABELS = len(labels)
id2label = {i:l for i,l in enumerate(labels)}
label2id = {l:i for i,l in enumerate(labels)}
data["labels"] = data.category.map(lambda x: label2id[x.strip()])

SIZE = data.shape[0]

train_texts = list(data.text[:SIZE//2])
val_texts =   list(data.text[SIZE//2:(3*SIZE)//4 ])
test_texts =  list(data.text[(3*SIZE)//4:])

train_labels = list(data.labels[:SIZE//2])
val_labels = list(data.labels[SIZE//2:(3*SIZE)//4])
test_labels = list(data.labels[(3*SIZE)//4:])


# 单语言模型:BERT
from transformers import BertTokenizerFast, BertForSequenceClassification, AutoTokenizer

# tokenizer = BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased", max_length=512)
# model = BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased",
#                                                       num_labels=NUM_LABELS, 
#                                                       id2label=id2label, label2id=label2id)

# 多语言模型:mBERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
model.to(device)

# 跨语言模型:XLM-R
# from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
# model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base",num_labels=NUM_LABELS, id2label=id2label, label2id=label2id)
# model.to(device)


from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings  = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = MyDataset(train_encodings, train_labels)
val_dataset = MyDataset(val_encodings, val_labels)
test_dataset = MyDataset(test_encodings, test_labels)

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support 
def compute_metrics(pred): 
    labels = pred.label_ids 
    preds = pred.predictions.argmax(-1) 
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro') 
    acc = accuracy_score(labels, preds) 
    return { 
        'Accuracy': acc, 
        'F1': f1, 
        'Precision': precision, 
        'Recall': recall 
    } 

training_args = TrainingArguments(
    # The output directory where the model predictions and checkpoints will be written
    output_dir='./TTC4900Model', 
    do_train=True,
    do_eval=True,
    #  The number of epochs, defaults to 3.0 
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=32,
    # Number of steps used for a linear warmup
    warmup_steps=100,                
    weight_decay=0.01,
    logging_strategy='steps',
   # TensorBoard log directory                 
    logging_dir='./multi-class-logs',            
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="epoch", 
    fp16=True,
    load_best_model_at_end=True
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned 
    model=model,
     # training arguments that we defined above                        
    args=training_args,                 
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
    compute_metrics= compute_metrics
)

trainer.train()

q = [trainer.evaluate(eval_dataset=data) for data in [train_dataset, val_dataset, test_dataset]]
pd.DataFrame(q, index=["train","val","test"]).iloc[:,:5]