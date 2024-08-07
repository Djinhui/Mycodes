# 《自然语言处理实战：从入门到项目实践》 CH03
# 《自然语言处理实战：从入门到项目实践》CH03 https://github.com/practical-nlp/practical-nlp-code

# 1. 基本向量化
# One-Hot Encoding
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
vocab = {}
count = 0
for doc in processed_docs:
    for word in doc.split():
        if word not in vocab:
            count = count +1
            vocab[word] = count
print(vocab) # {'dog': 1, 'bites': 2, 'man': 3, 'eats': 4, 'meat': 5, 'food': 6}

#Get one hot representation for any string based on this vocabulary. 
#If the word exists in the vocabulary, its representation is returned. 
#If not, a list of zeroes is returned for that word. 
def get_onehot_vector(somestring):
    onehot_encoded = []
    for word in somestring.split():
        temp = [0]*len(vocab)
        if word in vocab:
            temp[vocab[word]-1] = 1 # -1 is to take care of the fact indexing in array starts from 0 and not 1
        onehot_encoded.append(temp)
    return onehot_encoded

print(processed_docs[1])
get_onehot_vector(processed_docs[1]) #one hot representation for a text from our corpus.

S1 = 'dog bites man'
S2 = 'man bites dog'
S3 = 'dog eats meat'
S4 = 'man eats food'

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

data = [S1.split(), S2.split(), S3.split(), S4.split()]
values = data[0]+data[1]+data[2]+data[3] 
print("The data: ",values) # ['dog', 'bites', 'man', 'man', 'bites', 'dog', 'dog', 'eats', 'meat', 'man', 'eats', 'food']

#Label Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print("Label Encoded:",integer_encoded)

#One-Hot Encoding
onehot_encoder = OneHotEncoder()
onehot_encoded = onehot_encoder.fit_transform(data).toarray()
print("Onehot Encoded Matrix:\n",onehot_encoded) # 对每一列的不同值Onehot，第一列2个不同值，第二列2个不同值， 第三列4个不同值， 总维度为2+2+4=8，而不是Vocab维度6
'''
Onehot Encoded Matrix:
 [[1. 0. 1. 0. 0. 0. 1. 0.]
 [0. 1. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 1. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 1. 0. 0.]]
'''

# Bag of Words
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]

from sklearn.feature_extraction.text import CountVectorizer

#look at the documents list
print("Our corpus: ", processed_docs)

count_vect = CountVectorizer()
#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_) # {'dog': 1, 'bites': 0, 'man': 4, 'eats': 2, 'meat': 5, 'food': 3}

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])
print("Bow representation for 'dog and dog are friends':", temp.toarray()) # [[0 2 0 0 0 0]]


#BoW with binary vectors
count_vect = CountVectorizer(binary=True)
count_vect.fit(processed_docs)
temp = count_vect.transform(["dog and dog are friends"])
print("Bow representation for 'dog and dog are friends':", temp.toarray()) #  [[0 1 0 0 0 0]]

# Bag of N-Grams
#Ngram vectorization example with count vectorizer and uni, bi, trigrams
count_vect = CountVectorizer(ngram_range=(1,3))

#Build a BOW representation for the corpus
bow_rep = count_vect.fit_transform(processed_docs)

#Look at the vocabulary mapping
print("Our vocabulary: ", count_vect.vocabulary_)
# Our vocabulary:  {'dog': 3, 'bites': 0, 'man': 12, 'dog bites': 4, 'bites man': 2, 'dog bites man': 5, 'man bites': 13, 'bites dog': 1, 'man bites dog': 14, 'eats': 8, 'meat': 17, 'dog eats': 6, 'eats meat': 10, 'dog eats meat': 7, 'food': 11, 'man eats': 15, 'eats food': 9, 'man eats food': 16}

#see the BOW rep for first 2 documents
print("BoW representation for 'dog bites man': ", bow_rep[0].toarray())
print("BoW representation for 'man bites dog: ",bow_rep[1].toarray())

#Get the representation using this vocabulary, for a new text
temp = count_vect.transform(["dog and dog are friends"])

print("Bow representation for 'dog and dog are friends':", temp.toarray())

# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
bow_rep_tfidf = tfidf.fit_transform(processed_docs)

#IDF for all words in the vocabulary
print("IDF for all words in the vocabulary",tfidf.idf_)
print("-"*10)
#All words in the vocabulary.
print("All words in the vocabulary",tfidf.get_feature_names())
print("-"*10)

#TFIDF representation for all documents in our corpus 
print("TFIDF representation for all documents in our corpus\n",bow_rep_tfidf.toarray()) 
print("-"*10)

temp = tfidf.transform(["dog and man are friends"])
print("Tfidf representation for 'dog and man are friends':\n", temp.toarray())


# 2. 分布式表示
# Word2Vec、Glove、Fasttext、Doc2Vec
# 预训练
gn_vec_path = 'GoogleNews-vectors-negative300.bin'
import warnings #This module ignores the various types of warnings generated
warnings.filterwarnings("ignore") 

import psutil #This module helps in retrieving information on running processes and system resource utilization
import os
process = psutil.Process(os.getpid())
from psutil import virtual_memory
mem = virtual_memory()

import time #This module is used to calculate the time 

from gensim.models import Word2Vec, KeyedVectors
pretrainedpath = gn_vec_path

#Load W2V model. This will take some time, but it is a one time effort! 
pre = process.memory_info().rss
print("Memory used in GB before Loading the Model: %0.2f"%float(pre/(10**9))) #Check memory usage before loading the model 0.17G
print('-'*10)

start_time = time.time() #Start the timer
ttl = mem.total #Toal memory available

w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True) #load the model
print("%0.2f seconds taken to load"%float(time.time() - start_time)) #Calculate the total time elapsed since starting the timer
print('-'*10)

print('Finished loading Word2Vec')
print('-'*10)

post = process.memory_info().rss
print("Memory used in GB after Loading the Model: {:.2f}".format(float(post/(10**9)))) #Calculate the memory used after loading the model 5.06G
print('-'*10)

print("Percentage increase in memory usage: {:.2f}% ".format(float((post/pre)*100))) #Percentage increase in memory after loading the model
print('-'*10)

print("Numver of words in vocablulary: ",len(w2v_model.vocab)) #Number of words in the vocabulary. 

#Let us examine the model by knowing what the most similar words are, for a given word!
w2v_model.most_similar('beautiful')

#What is the vector representation for a word? 
w2v_model['computer']

#What if I am looking for a word that is not in this vocabulary?
w2v_model['practicalnlp'] #KeyError: "word 'practicalnlp' not in vocabulary"

import spacy

nlp = spacy.load('en_core_web_md')
# process a sentence using the model
mydoc = nlp("Canada is a large country")
#Get a vector for individual words
#print(mydoc[0].vector) #vector for 'Canada', the first word in the text 
print(mydoc.vector) #Averaged vector for the entire sentence

temp = nlp('practicalnlp is a newword')
temp[0].vector # all 0

# 自训练
from gensim.models import Word2Vec
# define training data
#Genism word2vec requires that a format of ‘list of lists’ be provided for training where every document contained in a list.
#Every list contains lists of tokens of that document.
corpus = [['dog','bites','man'], ["man", "bites" ,"dog"],["dog","eats","meat"],["man", "eats","food"]]

# Training the model
model_cbow = Word2Vec(corpus, min_count=1,sg=0) #using CBOW Architecture for trainnig
model_skipgram = Word2Vec(corpus, min_count=1,sg=1)#using skipGram Architecture for training

#Summarize the loaded model
print(model_cbow)

#Summarize vocabulary
words = list(model_cbow.wv.vocab)
print(words)

#Acess vector for one word
print(model_cbow['dog'])

#Compute similarity 
print("Similarity between eats and bites:",model_cbow.similarity('eats', 'bites'))
print("Similarity between eats and man:",model_cbow.similarity('eats', 'man'))

#Most similarity
model_cbow.most_similar('meat')

# save model
model_cbow.save('model_cbow.bin')

# load model
new_model_cbow = Word2Vec.load('model_cbow.bin')
print(new_model_cbow)

file_name = "data/en/enwiki-latest-pages-articles-multistream14.xml-p13159683p14324602.bz2"
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
import time
# from gensim.models.glove import Glove

#Preparing the Training data
wiki = WikiCorpus(file_name, lemmatize=False, dictionary={})
sentences = list(wiki.get_texts())

#if you get a memory error executing the lines above
#comment the lines out and uncomment the lines below. 
#loading will be slower, but stable.
# wiki = WikiCorpus(file_name, processes=4, lemmatize=False, dictionary={})
# sentences = list(wiki.get_texts())

#if you still get a memory error, try settings processes to 1 or 2 and then run it again.

#CBOW
start = time.time()
word2vec_cbow = Word2Vec(sentences,min_count=10, sg=0)
end = time.time()

print("CBOW Model Training Complete.\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0)) # 0.07h

#Summarize the loaded model
print(word2vec_cbow)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_cbow.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_cbow['film'])}")
print(word2vec_cbow['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_cbow.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_cbow.similarity('film', 'tiger'))
print("-"*30)

# save model
from gensim.models import Word2Vec, KeyedVectors   
word2vec_cbow.wv.save_word2vec_format('word2vec_cbow.bin', binary=True)

# load model
new_modelword2vec_cbow = Word2Vec.load('word2vec_cbow.bin')

#SkipGram
start = time.time()
word2vec_skipgram = Word2Vec(sentences,min_count=10, sg=1)
end = time.time()

print("SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0)) # 0.2H
#Summarize the loaded model
print(word2vec_skipgram)
print("-"*30)

#Summarize vocabulary
words = list(word2vec_skipgram.wv.vocab)
print(f"Length of vocabulary: {len(words)}")
print("Printing the first 30 words.")
print(words[:30])
print("-"*30)

#Acess vector for one word
print(f"Length of vector: {len(word2vec_skipgram['film'])}")
print(word2vec_skipgram['film'])
print("-"*30)

#Compute similarity 
print("Similarity between film and drama:",word2vec_skipgram.similarity('film', 'drama'))
print("Similarity between film and tiger:",word2vec_skipgram.similarity('film', 'tiger'))
print("-"*30)

# save model
word2vec_skipgram.wv.save_word2vec_format('word2vec_sg.bin', binary=True)

# load model
new_model_skipgram = Word2Vec.load('model_skipgram.bin')

# FastText
from gensim.models.fasttext import FastText
fasttext_cbow = FastText(sentences, sg=0, min_count=10)
end = time.time()

print("FastText CBOW Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0)) # 0.23h

#SkipGram
start = time.time()
fasttext_skipgram = FastText(sentences, sg=1, min_count=10)
end = time.time()

print("FastText SkipGram Model Training Complete\nTime taken for training is:{:.2f} hrs ".format((end-start)/3600.0)) # 0.34

import fasttext
'''
fastText文本分类要求的数据存储格式：
__label__1	面 积 不 大 的 水 塘 真 藏 货 ， 阿 琪 抽 水 抗 旱 还 有 意 外 收 获 ， 抓 鱼 抓 得 过 瘾
__label__1	广 安 市 群 策 群 力 抗 旱
__label__2	湖 北 省 《 开 发 性 金 融 支 持 县 域 垃 圾 污 水 处 理 设 施 建 设 实 施 细 则 》
__label__2	旱 灾 究 竟 有 多 恐 怖 ？ 很 多 人 都 不 理 解 为 何 路 边 的 树 都 没 了 皮
__label__0	钱 铺 镇 多 措 并 举 抗 旱 情
__label__2	夷 陵 消 防 获 赠 锦 旗 ： 抗 旱 为 民 办 实 事 ， 锦 旗 诠 释 “ 鱼 水 情 ”
代码：
"__label__"+str(label)+"\t"+" ".join(segs)
'''
classifier = fasttext.train_supervised('./data/train_data.txt',label='__label__', wordNgrams=2,epoch=20,lr=0.1,dim=100)
 
#参数说明
'''
train_supervised(input, lr=0.1, dim=100, 
                   ws=5, epoch=5, minCount=1, 
                   minCountLabel=0, minn=0, 
                   maxn=0, neg=5, wordNgrams=1, 
                   loss="softmax", bucket=2000000, 
                   thread=12, lrUpdateRate=100,
                   t=1e-4, label="__label__", 
                   verbose=2, pretrainedVectors="")
'''

#Skipgram Model 
model = fasttext.train_unsupervised('data.txt', model = 'skipgram')

#Cbow Model
model = fasttext.train_unsupervised('data.txt', model = 'cbow')

# Doc2Vec:
# Doc2vec allows us to directly learn the representations for texts of arbitrary lengths (phrases, sentences, paragraphs and documents), 
# by considering the context of words in the text into account.
import spacy

# 预训练的Doc2Vec
nlp = spacy.load('en_core_web_sm') # here nlp object refers to the 'en_core_web_sm' language model instance.

#Assume each sentence in documents corresponds to a separate document.
documents = ["Dog bites man.", "Man bites dog.", "Dog eats meat.", "Man eats food."]
processed_docs = [doc.lower().replace(".","") for doc in documents]
processed_docs

print("Document After Pre-Processing:",processed_docs)


#Iterate over each document and initiate an nlp instance.
for doc in processed_docs:
    doc_nlp = nlp(doc) #creating a spacy "Doc" object which is a container for accessing linguistic annotations. 
    
    print("-"*30)
    print("Average Vector of '{}'\n".format(doc),doc_nlp.vector)#this gives the average vector of each document
    for token in doc_nlp:
        print()
        print(token.text,token.vector)#this gives the text of each word in the doc and their respective vectors.

# 训练Doc2Vec
import warnings
warnings.filterwarnings('ignore')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pprint import pprint
import nltk
nltk.download('punkt')

data = ["dog bites man",
        "man bites dog",
        "dog eats meat",
        "man eats food"]

tagged_data = [TaggedDocument(words=word_tokenize(word.lower()), tags=[str(i)]) for i, word in enumerate(data)]
'''
[TaggedDocument(words=['dog', 'bites', 'man'], tags=['0']),
 TaggedDocument(words=['man', 'bites', 'dog'], tags=['1']),
 TaggedDocument(words=['dog', 'eats', 'meat'], tags=['2']),
 TaggedDocument(words=['man', 'eats', 'food'], tags=['3'])]
'''
#dbow
model_dbow = Doc2Vec(tagged_data,vector_size=20, min_count=1, epochs=2,dm=0)
print(model_dbow.infer_vector(['man','eats','food']))#feature vector of man eats food
model_dbow.wv.most_similar("man",topn=5)#top 5 most simlar words.
model_dbow.wv.n_similarity(["dog"],["man"])

#dm
model_dm = Doc2Vec(tagged_data, min_count=1, vector_size=20, epochs=2,dm=1)
print("Inference Vector of man eats food\n ",model_dm.infer_vector(['man','eats','food']))
print("Most similar words to man in our corpus\n",model_dm.wv.most_similar("man",topn=5))
print("Similarity between man and dog: ",model_dm.wv.n_similarity(["dog"],["man"]))

# 可视化词嵌入-T-SNE
from gensim.models import Word2Vec, KeyedVectors #To load the model
import warnings
warnings.filterwarnings('ignore') #ignore any generated warnings
import numpy as np
import matplotlib.pyplot as plt #to generate the t-SNE plot
from sklearn.manifold import TSNE #scikit learn's TSNE 
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

model = KeyedVectors.load_word2vec_format('word2vec_cbow.bin',binary=True)

#Preprocessing our models vocabulary to make better visualizations
words_vocab= list(model.wv.vocab)#all the words in the vocabulary. 
print("Size of Vocabulary:",len(words_vocab))
print("Few words in Vocabulary",words_vocab[:50])

#Let us remove the stop words from this it will help making the visualization cleaner
stopwords_en = stopwords.words()
words_vocab_without_sw = [word.lower() for word in words_vocab if not word in stopwords_en]
print("Size of Vocabulary without stopwords:",len(words_vocab_without_sw))
print("Few words in Vocabulary without stopwords",words_vocab_without_sw[:30])
#The size didnt reduce much after removing the stop words so lets try visualizing only a selected subset of words

#With the increase in the amount of data, it becomes more and more difficult to visualize and interpret
#In practice, similar words are combined into groups for further visualization.

keys = ['school', 'year', 'college', 'city', 'states', 'university', 'team', 'film']
embedding_clusters = []
word_clusters = []

for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)#apending access vector of all similar words
    word_clusters.append(words)#appending list of all smiliar words

from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape #geting the dimensions
tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2) #reshaping it into 2d so we can visualize it

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

#script for constructing two-dimensional graphics using Matplotlib
def tsne_plot_similar_words(labels, embedding_clusters, word_clusters, a=0.7):
    plt.figure(figsize=(16, 9))
    

    for label, embeddings, words in zip(labels, embedding_clusters, word_clusters):
        x = embeddings[:,0]
        y = embeddings[:,1]
        plt.scatter(x, y, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2), 
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.grid(True)
    plt.show()

tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)


tsne_model_en_2d = TSNE(perplexity=5, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)

tsne_model_en_2d = TSNE(perplexity=25, n_components=2, init='pca', n_iter=1500, random_state=2020) 
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
tsne_plot_similar_words(words_vocab_without_sw, embeddings_en_2d, word_clusters)

# 可视化词嵌入-TensorBoard
#making the required imports
import warnings #ignoring the generated warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
from gensim.models import KeyedVectors
import os

#Loading the model
cwd=os.getcwd() 
model = KeyedVectors.load_word2vec_format(cwd+'\Models\word2vec_cbow.bin', binary=True)
#get the model's vocabulary size
max_size = len(model.wv.vocab)-1
#make a numpy array of 0s with the size of the vocabulary and dimensions of our model
w2v = np.zeros((max_size,model.wv.vector_size))

#Now we create a new file called metadata.tsv where we save all the words in our model 
#we also store the embedding of each word in the w2v matrix
if not os.path.exists('projections'):
    os.makedirs('projections')
    
with open("projections/metadata.tsv", 'w+',encoding="utf-8") as file_metadata: #changed    added encoding="utf-8"
    for i, word in enumerate(model.wv.index2word[:max_size]):
        
        #store the embeddings of the word
        w2v[i] = model.wv[word]
        #write the word to a file 
        file_metadata.write(word + '\n')

#initializing tf session
sess = tf.InteractiveSession()
#Initialize the tensorflow variable called embeddings that holds the word embeddings:
with tf.device("/cpu:0"):
    embedding = tf.Variable(w2v, trainable=False, name='embedding')

#Initialize all variables
tf.global_variables_initializer().run()

#object of the saver class which is actually used for saving and restoring variables to and from our checkpoints
saver = tf.train.Saver()

#with FileWriter,we save summary and events to the event file
writer = tf.summary.FileWriter('projections', sess.graph)

# Initialize the projectors and add the embeddings
config = projector.ProjectorConfig()
embed= config.embeddings.add()

#specify our tensor_name as embedding and metadata_path to the metadata.tsv file
embed.tensor_name = 'embedding'
embed.metadata_path = 'metadata.tsv'

#save the model
projector.visualize_embeddings(writer, config)
saver.save(sess, 'projections/model.ckpt', global_step=max_size)

# in cmd: tensorboard --logdir=projections --port=8000
# 3. 通用文本表示
# ELMo、BERT
# 4. 人工特征表示
