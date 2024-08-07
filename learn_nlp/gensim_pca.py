# 《精通Transformer》CH02

import nltk
from nltk.corpus import gutenberg
from gensim.models.word2vec import Word2Vec
from gensim.models import FastText
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

nltk.download('gutenberg')
macbeth = gutenberg.sents('shakespeare-macbeth.txt')

'''
from gensim.models import Word2Vec, KeyedVectors
pretrainedpath = "NLPBookTut/GoogleNews-vectors-negative300.bin"
w2v_model = KeyedVectors.load_word2vec_format(pretrainedpath, binary=True) # gensim还支持训练和加载GloVe预训练模型，gensim封装了fasttext
print('done loading Word2Vec')
print(len(w2v_model.vocab)) # 词汇表中词的数量
print(w2v_model.most_similar('beautiful'))
print('beautiful representation:', w2v_model['beautiful'])
'''


def plotWords3D(vecs, words, title):
    """
        Parameters
        ----------
        vecs : numpy-array
            Transformed 3D array either by PCA or other techniques
        words: a list of word
            the word list to be mapped
        title: str
            The title of plot     
        """
    fig = plt.figure(figsize=(14,10))
    ax = fig.gca(projection='3d')
    for w, vec in zip(words, vecs):
        ax.text(vec[0],vec[1],vec[2], w, color=np.random.rand(3,))
    ax.set_xlim(min(vecs[:,0]), max(vecs[:,0]))
    ax.set_ylim(min(vecs[:,1]), max(vecs[:,1]))
    ax.set_zlim(min(vecs[:,2]), max(vecs[:,2]))
    ax.set_xlabel('DIM-1')
    ax.set_ylabel('DIM-2')
    ax.set_zlabel('DIM-3')
    plt.title(title)
    plt.show()

model = Word2Vec(sentences=macbeth, size=100, window=5, min_count=10, workers=4, iter=100)
# model.wv.most_similar('hate')
words = list([e for e in model.wv.vocab])
words3d = PCA(n_components=3).fit_transform(model.wv[words[:100]])
plotWords3D(words3d, words, "Visualizing Word2Vec Word Embeddings using PCA")

model = FastText(sentences=macbeth, size=100, window=5, min_count=10, workers=4, iter=100, word_ngrams=3)

words = [w[0] for w in model.wv.similar_by_word('Macbeth', 50)]
words3d = PCA(n_components=3).fit_transform(model.wv[words])
plotWords3D(words3d, words, "Visualizing FastText Word Embeddings using PCA")