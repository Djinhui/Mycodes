# -*- coding:utf-8 -*-
# @Time: XX:XX
# @Author: XXX
# @File: word2vec_gensim.py
# @Software: PyCharm
# @address:https://mp.weixin.qq.com/s/krpBy8MeXX7twtCDEXsWhQ
import jieba
import re
from gensim.models import word2vec


def read_stop():
    stop_words = []
    with open("data\stop_words.txt", "r", encoding="utf-8") as f_reader:
        for line in f_reader:
            line = line.replace("\r", "").replace("\n", "").strip()
            stop_words.append(line)
    print(len(stop_words))
    stop_words = set(stop_words)
    print(len(stop_words))
    return stop_words

def data_process():
    sentecnces = []
    rules = u"[\u4e00-\u9fa5]+"
    pattern = re.compile(rules)
    f_writer = open("data\分词后的天龙八部.txt", "w", encoding="utf-8")

    with open("data\天龙八部.txt", "r", encoding="utf-8") as f_reader:
        for line in f_reader:
            line = line.replace("\r", "").replace("\n", "").strip()
            if line == "" or line is None:
                continue
            line = " ".join(jieba.cut(line))
            seg_list = pattern.findall(line)
            word_list = []
            for word in seg_list:
                if word not in stop_words:
                    word_list.append(word)
            if len(word_list) > 0:
                sentecnces.append(word_list)
                line = " ".join(word_list)
                f_writer.write(line + "\n")
                f_writer.flush()
    f_writer.close()
    return sentecnces


if __name__ == '__main__':
    stop_words = read_stop()
    sentecnces = data_process()
    model = word2vec.Word2Vec(sentecnces, size=100, window=5, min_count=1, workers=4, sg=0)

    # 保存模型
    model.save("data\天龙八部.bin")
    # 加载模型
    model = word2vec.Word2Vec.load("data\天龙八部.bin")

    # 选择和乔峰相近的前5个词
    for e in model.wv.most_similar(positive=['乔峰'], topn=5):
        print(e[0], e[1])

    # 计算两个词的相似度/相关程度
    sim_value = model.wv.similarity('乔峰', '虚竹')
    print(sim_value)

    # 计算两个集合的相似度/相关程度
    sim_value = model.wv.n_similarity(['乔峰', '虚竹'], ['乔峰', '段誉'])
    print(sim_value)

    # 查看词向量
    print(type(model.wv['乔峰'])) # <class 'numpy.ndarray'>
    print(len(model.wv['乔峰']))  # 100
    print(model.wv['乔峰']) 

    