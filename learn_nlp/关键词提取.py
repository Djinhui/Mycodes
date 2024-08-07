import os
import csv


# 1. ===============data_prepare.py======================
# 将多个txt文件合并为单个csv文件,每个txt文件的文件名为id_标题.txt，文件内容， csv文件有三列，[id， 标题， 摘要]
def text_combine(path):
    files = []
    for file in os.listdir(path):
        if file.endswith('.txt'):
            files.append(path + "/" + file)
    with open('text.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'title', 'abstract'])
        for file in files:
            number = file.split('/')[-1].split('_')[0] # id
            title, text = '', ''
            count = 0
            with open(file, 'r', encoding='utf-8-sig') as f1:
                for line in f1:
                    if count == 0:
                        title += line.strip()
                    else:
                        text += line.strip()
                    count += 1
            writer.writerow([number, title, text])
    print('text.csv done')


# 2.1 ==============tfidf.py基于TF-IDF的关键词提取=============
import codecs
import pandas as pd
import jieba.posseg
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def data_read(text, stopkey):
    l =[]
    pos = ['n', 'nz','v', 'vd', 'vn', 'l', 'a','d'] # 定义选取的词性
    seg = jieba.posseg.cut(text)
    for i in seg:
        if i.word not in stopkey and i.flag in pos:
            l.append(i.word)
    return l

def words_tfidf(data, stopkey, topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    corpus = [] # 将所以文档输出到一个List中，一行为一个文档
    for index in range(len(idList)):
        # 拼接标题和摘要
        text = titleList[index] + '。' + abstractList[index]
        text = data_read(text, stopkey)
        text = " ".join(text)
        corpus.append(text)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    weight = tfidf.toarray() # tfidf矩阵，weight[i,j]表示j词在i文本中的tfidf权重
    words = vectorizer.get_feature_names() # 词袋模型中所有词
    
    ids, titles, keys = [], [], []
    for i in range(len(weight)):
        ids.append(idList[i])
        titles.append(titleList[i])
        df_word, df_wight = [], []
        for j in range(len(words)):
            df_word.append(words[j])
            df_wight.append(weight[i][j])
        word_weight = pd.DataFrame({'word': df_word, 'weight': df_wight})
        word_weight = word_weight.sort_values(by='weight', ascending=False)
        key =' '.join(word_weight['word'].head(topK))
        keys.append(key.encode('utf-8').decode('utf-8'))

    result = pd.DataFrame({'id': ids, 'title': titles, 'keywords': keys})
    return result


data_file = 'text.csv'
data = pd.read_csv(data_file)
stopkey = [w.strip() for w in codecs.open('stopwords.txt', 'r', 'utf-8').readlines()]
result = words_tfidf(data, stopkey, 10)
result.to_csv('tfidf.csv', index=False)



# 2.2 ==============textrank.py基于textrank的关键词提取=============
import jieba.analyse


def words_textrank(data, topK):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    
    ids, titles, keys = [], [], []
    for index in range(len(idList)):
        # 拼接标题和摘要
        text = titleList[index] + '。' + abstractList[index]
        jieba.analyse.set_stop_words('stopwords.txt') # 加载自定义停用此表
        # TextRank关键词提取，词性筛选
        keywords = jieba.analyse.textrank(text, topK=topK,
                                          allowPOS=('n', 'nz','v', 'vd', 'vn', 'l', 'a','d'))
        key =' '.join(keywords)
        ids.append(idList[index])
        titles.append(titleList[index])
        keys.append(key.encode('utf-8').decode('utf-8'))

    result = pd.DataFrame({'id': ids, 'title': titles, 'keywords': keys})
    return result

data_file = 'text.csv'
data = pd.read_csv(data_file)
result = words_textrank(data, 10)
result.to_csv('textrank.csv', index=False)


# 2.3.1 ==============word2vec_prepare.py构建候选词向量=============
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import codecs
import pandas as pd
import numpy as np
import jieba
import jieba.posseg
import gensim


def data_prepare(text, stopkey):
    l =[]
    pos = ['n', 'nz','v', 'vd', 'vn', 'l', 'a','d'] # 定义选取的词性
    seg = jieba.posseg.cut(text)
    for i in seg:
        if i.word not in stopkey and i.flag in pos and i.word not in l:
            l.append(i.word)
    return l


def word_vecs(wordList, model):
    name = []
    vecs = []
    for word in wordList:
        word = word.replace('\n', '')
        try:
            if word in model:
                name.append(word.encode('utf-8').decode('utf-8'))
                vecs.append(model[word])
        except KeyError:
            continue
    a = pd.DataFrame(name, columns=['word'])
    b = pd.DataFrame(np.array(vecs, dtype='float'))
    c = pd.concat([a, b], axis=1)
    return c


def build_words_vecs(data, stopkey, model):
    idList, titleList, abstractList = data['id'], data['title'], data['abstract']
    for index in range(len(idList)):
        id_ = idList[index]
        title = titleList[index]
        abstract = abstractList[index]
        l_ti = data_prepare(title, stopkey) # 处理标题
        l_ab = data_prepare(abstract, stopkey) # 处理摘要

        words = np.append(l_ti, l_ab)
        words = list(set(words)) # 去重
        wordvecs = word_vecs(words, model)

        data_vecs = pd.DataFrame(wordvecs)
        data_vecs.to_csv('result/vecs/wordvecs_' + str(id_) + '.csv', index=False)
        print('第' + str(index) + '个文件处理完成')

        

data_file = 'text.csv'
data = pd.read_csv(data_file)
stopkey = [w.strip() for w in codecs.open('stopwords.txt', 'r', 'utf-8').readlines()]
inp = 'wiki.zh.text.vector' # 基于wiki中文训练好的词向量
model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
build_words_vecs(data, stopkey, model)


# 2.3.2 ==============word2vec_result.py构建基于W2V词聚类的关键词提取=============
import os
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import math



def words_kmeans(data:pd.DataFrame, topK:int=10):
    words = data['word']
    vecs = data.iloc[:, 1:]
    kmeans = KMeans(n_clusters=1, random_state=0).fit(vecs)
    labels = kmeans.labels_
    labels = pd.DataFrame(labels, columns=['label'])
    new_df = pd.concat([labels, vecs], axis=1)
    vec_center = kmeans.cluster_centers_

    distances = []
    vec_words = np.array(vecs)
    vec_center = vec_center[0] # 第一个类别聚类中心,本例只有一个类别
    for i in range(len(vec_words)):
        distance = math.sqrt(np.sum((vec_words[i] - vec_center)**2))
        distances.append(distance)
    distances = pd.DataFrame(distances, columns=['distance'])
    result = pd.concat([words, labels, distances], axis=1)
    result = result.sort_values(by='distance', ascending=True)

    wordlist = np.array(result['word'])
    wordlist = wordlist[:topK]
    wordlist = " ".join(wordlist)
    return wordlist

dataFile = 'text.csv'
articleData = pd.read_csv(dataFile)
ids, titles, keys = [], [], []
rootdir = 'result/vecs' # 2.3.1提取的各文件词向量目录
fileList = os.listdir(rootdir)

for i in range(len(fileList)):
    filename = fileList[i]
    path = os.path.join(rootdir, filename)
    if os.path.isfile(path):
        data = pd.read_csv(path) # 读取词向量文件数据
        artile_keys = words_kmeans(data, 10)
        shortname, extension = os.path.splitext(filename)
        t = shortname.split('_')
        article_id = int(t[len(t)-1])
        artile_title = articleData[articleData['id'] == article_id]['title'].values[0]
        ids.append(article_id)
        titles.append(artile_title)
        keys.append(artile_keys.encode('utf-8').decode('utf-8'))

result = pd.DataFrame({'id':ids, 'title':titles, 'key':keys})
result.to_csv('word2vec.csv', index=False, encoding='utf-8-sig')