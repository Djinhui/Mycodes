import logging
import os.path
import sys
from gensim.corpora import WikiCorpus

# ===============1.process_wiki_data.py ====================
# sys.args[0]获取的是脚本文件的文件名称
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" %''.join(sys.argv))

if len(sys.argv) < 3:
    print(globals()['__doc__'] % locals())
    sys.exit(1)
in_f = sys.argv[1]
out_f = sys.argv[2]
space = " "
i = 0

output = open(out_f, 'w', encoding='utf-8')
wiki = WikiCorpus(in_f, lemmatize=False, dictionary={})
for text in wiki.get_texts():
    output.write(space.join(text) + "\n")
    i = i + 1
    if (i % 10000 == 0):
        logger.info("Saved " + str(i) + " articles")
output.close()
logger.info("Finished Saved " + str(i) + " articles")

# in cmd window: python process_wiki_data.py articles.xml.bz.txt wiki.zh.txt

# ===============2.seg.py ====================
import jieba
import jieba.analyse
import codecs

def process_wiki_text(org_file, target_file):
    with codecs.open(org_file, 'r', 'utf-8') as inp, codecs.open(target_file, 'w', 'utf-8') as outp:
        line = inp.readline()
        line_num = 1
        while line:
            print('---- processing ', line_num,'article----------------')
            line_seg = jieba.cut(line)
            outp.writelines(" ".join(line_seg))
            line_num = line_num + 1
            line = inp.readline()
    
    inp.close() 
    outp.close()
    print('well done.')

process_wiki_text('wiki.zh.txt', 'wiki.zh.text.seg')



# =====================train_word2vec_model.py=================
import logging
import os.path
import sys  
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence # 读取分词文件

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" %''.join(sys.argv))

if len(sys.argv) < 4:
    print(globals()['__doc__'] % locals())
    sys.exit(1)

# sys.args[0]获取的是脚本文件的文件名称
inp = sys.argv[1] # 分词后的文件
outp1 = sys.argv[2] # 训练好的模型
outp2 = int(sys.argv[3]) # 得到的词向量

model = Word2Vec(LineSentence(inp), size=100, window=5, min_count=5, 
                 workers=multiprocessing.cpu_count())
# 保存模型，供日後使用
model.save(outp1)
# 保存词向量
model.wv.save_word2vec_format(outp2, binary=False)

# in cmd:python train_word2vec_model.py wiki.zh.text.seg wiki.zh.text.model wiki.zh.text.vector

# ======================gensim_test.py=======================
from gensim.models import Word2Vec

# 加载模型
model = Word2Vec.load("wiki.zh.text.model")
# 计算两个词的相似度/相关程度
# y1 = model.similarity("woman", "man")
count = 0
for word in model.wv.index2word:
    print(word, model[word]) # model[word]返回word的词向量
    count += 1
    if count == 10:
        break
# 
res = model.most_similar(u"男人")
for e in res:
    print(e)