# pip install glove_python

from glove import Glove
from glove import Corpus

sentense = [['你','是','谁'],['我','是','中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)


glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

glove.save('glove.model')
glove = Glove.load('glove.model')

corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')

glove.most_similar('我', number=10)

# 全部词向量矩阵
glove.word_vectors
# 指定词条词向量
glove.word_vectors[glove.dictionary['你']]


corpus_model.matrix.todense().tolist()


