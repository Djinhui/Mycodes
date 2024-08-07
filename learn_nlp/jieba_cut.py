import jieba
jieba.enable_paddle()

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


'''
精确模式能获得句子的语义信息,因此自然语言处理的各种任务常常使用精确模式。
全模式和搜索引擎模式适用于搜索和推荐领域,
paddle模式则和精确模式类似,不同之处在于paddle模式匹配会对包含语义最大的词组进行切分
'''
def word_segment():
    strs = '我来到北京清华大学'
    seg_list = jieba.cut(strs, cut_all=True)
    print("Full Mode: " + "/ ".join(seg_list))  # 全模式
    seg_list = jieba.cut(strs, cut_all=False)
    print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
    seg_list = jieba.cut_for_search(strs)  # 搜索引擎模式
    print("Search Engine Mode: " + "/ ".join(seg_list))
    seg_list = jieba.cut(strs,use_paddle=True)  
    print("Paddle Mode: " + "/ ".join(seg_list))


def stopwordslist():
    stopwords = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strrip())
    stopwords = stopwordslist('./stopwords.txt')  # 这里加载停用词的路径
    outstr = ''
    for word in sentence_seged:
        if word not in stopwords:
            if word!= '\t':
                outstr += word
                outstr += " "
    return outstr

inputs = open('./data/comment.txt', 'r', encoding='gbk')
outputs = open('./data/comment_seg.txt', 'w', encoding='utf-8')
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg + '\n')
outputs.close()
inputs.close()

mask_image = np.array(Image.open('./data/heart.jpg'))

with open('./data/comment_seg.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    wordcloud = WordCloud(
        font_path='./data/simhei.ttf',
        background_color='white',
        mask=mask_image,
        max_words=2000,
        max_font_size=100).generate(text)
    
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    wordcloud.to_file('./data/wordcloud.jpg')


if __name__ == '__main__':
    word_segment()