# 1. 数据处理函数
import re
import codecs
import pickle
import collections
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

'''
renmin.txt
/w为标点符号标记
19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ——/w  一九九八年/t  新年/t  讲话/n  （/w  附/v  图片/n  １/m  张/q  ）/w  
19980101-02-003-001/m  党中央/nt  国务院/nt  关心/v  西藏/ns  雪灾/n  救灾/vn  工作/vn  
19980101-02-003-002/m  灾区/n  各级/r  政府/n  全力/n  组织/v  抗灾/v  力争/v  降低/v  灾害/n  损失/n  

'''

def origin_handle_entities():
    """
    将文本中分开的机构名和人名合并
    """
    with open('renmin.txt', 'r', encoding='utf-8') as inp, open('renmin2.txt', 'w', encoding='utf-8') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i < len(line) - 1:
                # 合并机构名称
                #[国务院/nt  侨办/j]/nt --> 国务院侨办/nt
                if line[i][0] == '[':
                    outp.write(line[i].split('/')[0][1:])
                    i += 1
                    while i < len(line)-1 and line[i].find(']') == -1:
                        if line[i] != '':
                            outp.write(line[i].split('/')[0])
                        i += 1
                    outp.write(line[i].split('/')[0].strip() + '/' + line[i].split('/')[1][-2:] + ' ')
                # 合并人名 裴/nr 广战/nr -->裴广战/nr
                elif line[i].split('/')[1] == 'nr':
                    word = line[i].split('/')[0]
                    i+= 1
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                        outp.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i] + '/nr ')
                i += 1
            outp.write('\n')

def origin_handle_mark():
    """
    标注数据的命名实体
    裴/B_nr
    广/M_nr
    战/E_nr
    """
    with open('renmin2.txt', 'r', encoding='utf-8') as inp, open('renmin3.txt', 'w', encoding='utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                # 处理人名、地名、机构名
                if tag in ['nr', 'ns', 'nt']:
                    outp.write(word[0] + '/B_' + tag +' ') 
                    for j in range(1, len(word)-1):
                        if j != ' ':
                            outp.write(word[j] + '/M_' + tag +' ')   
                    outp.write(word[-1] + '/E_' + tag +' ')
                # 处理其他
                else:
                    for wor in word:
                        outp.write(word + '/O')
                i += 1
            outp.write('\n')

def sentence_split():
    """
    按句分割
    """
    with open('renmin3.txt', 'r', encoding='utf-8') as inp, open('renmin4.txt', 'w', encoding='utf-8') as outp:
        texts = inp.read().encode('utf-8').decode('utf-8-sig')
        sentences = re.split('[，。！？、‘’“”]/[O]'.encode('utf-8').decode('utf-8'), texts)
        for sentence in sentences:
            outp.write(sentence.strip() + '\n')

def data_to_pkl():
    datas = []
    labels = []
    all_words = []
    tags = set()
    input_data = codecs.open('renmin4.txt', 'r', encoding='utf-8')
    # 将标注子句拆分成字列表和对应的标注列表
    for line in input_data.readlines():
        linedata = list()
        linelabel = list()
        line = line.split()
        numNot0 = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            all_words.append(word[0])
            tags.add(word[1])

            if word[1]!= 'O':
                numNot0 += 1
        if numNot0!= 0:
            datas.append(linedata)
            labels.append(linelabel)
    input_data.close()

    # 创建词汇表和标签表
    words_count = collections.Counter(all_words).most_common()
    word2id = {word:i for i, (word, _) in enumerate(words_count, 1)}
    word2id["[PAD]"] = 0
    word2id["[unknown]"] = len(word2id)
    id2word = {i:word for word, i in word2id.items()}
    tag2id = {tag:i for i, tag in enumerate(tags)}
    id2tag = {i:tag for tag, i in tag2id.items()}
    
    # 数据向量化
    max_len = 60
    data_ids = [[word2id[w] for w in line] for line in datas]
    label_ids =  [[tag2id[t] for t in line] for line in labels]
    x = pad_sequences(data_ids, maxlen=max_len, padding='post').astype(np.int64)
    y = pad_sequences(label_ids, maxlen=max_len, padding='post').astype(np.int64)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # save
    with open('renmindata.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)
        pickle.dump(x_train, outp)
        pickle.dump(y_train, outp)
        pickle.dump(x_test, outp)
        pickle.dump(y_test, outp)
        pickle.dump(x_valid, outp)
        pickle.dump(y_valid, outp)

def load_data_rm():
    pickle_path = 'renmindata.pkl'
    with open(pickle_path, 'rb') as inp:
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    with open('vocab.pkl', 'wb') as outp:
        pickle.dump(word2id, outp)
        pickle.dump(id2word, outp)
    with open('tags.pkl', 'wb') as outp:
        pickle.dump(tag2id, outp)
        pickle.dump(id2tag, outp)



def read_Word2vec():
    with open('vocab.pkl', 'rb') as inpp:
        token2idx = pickle.load(inp)
        idx2token = pickle.load(inp)
    with open('tags.pkl', 'rb') as inp:
        tag2idx = pickle.load(inp)
        idx2tag = pickle.load(inp)
    return token2idx, idx2token, tag2idx, idx2tag


def train_hmm(data_file):
    input_data = codecs.open(data_file, 'r', encoding='utf-8') # renmin4.txt
    pi = np.zeros(N) # 每个tag出现在句首的概率
    A = np.zeros((N,M)) # A[i][j]给定tag i出现单词j的概率
    B = np.zeros((N,N)) # B[i][j]词性为tag i时 其后单词词性为tag j的概率
    for line in input_data.readlines():
        # do something
        pass
    return pi, A, B

def save_hmm():
    file = 'renmin4.txt'
    pi,A,B = train_hmm(file)
    with open('hmm.pkl', 'wb') as f:
        pickle.dump(pi, f)
        pickle.dump(A, f)
        pickle.dump(B, f)
    return pi, A, B


def log_v(v):
    return np.log(v+1e-6)

def viterbi_decode(x, pi, A, B):
    T = len(x) # 待预测文本长度
    N = len(tag2idx) # 标签数据量
    dp = np.full((T,N), float('-inf'))
    ptr =np.zeros_like(dp, dtype=np.int32)
    dp[0] = log_v(pi) + log_v(A[:, x[0]])
    # 动态规划实现viterbi
    for i in range(1, T):
        v = dp[i-1].reshape(-1,1) + log_v(B)
        dp[i] = np.max(v, axis=0) + log_v(A[:, x[i]])
        ptr[i] = np.argmax(v, axis=0)

    best_seq = [0] * T
    best_seq[-1] = np.argmax(dp[-1])
    for i in range(T-2, -1,-1):
        best_seq[i] = ptr[i+1][best_seq[i+1]]

    return best_seq


class Tokenizer:
    def __init__(self, vocab_file):
        with open(vocab_file) as inp:
            self.token2idx = pickle.load(inp)
            self.idx2token = pickle.load(inp)
    
    def encode(self, text, maxlen):
        seqs = re.split('[，。！？、‘’“”]', text.strip())
        seq_ids = []
        for seq in seqs:
            token_ids = []
            if seq:
                for char in seq:
                    if char not in self.token2idx:
                        token_ids.append(self.token2idx['[unknown]'])
                    else:
                        token_ids.append(self.token2idx[char])
        num_samples = len(seq_ids)
        x = np.full((num_samples, maxlen), 0, dtype=np.int64)
        for idx, s in enumerate(seq_ids):
            trunc = np.array(s[:maxlen], dtype=np.int64)
            x[idx, :len(trunc)] = trunc
        return x
    
def predict(inputs_ids):
    res = []
    for idx, x in enumerate(inputs_ids):
        x = x[x>0]
        y_pred = viterbi_decode(x, pi, A, B)
        res.append(y_pred)
    return res


class Parser:
    def __init__(self, tags_file) -> None:
        with open(tags_file, 'rb') as f:
            self.tag2idx = pickle.load(f)
            self.idx2tag = pickle.load(f)

    def decode(self, text, paths):
        seqs = re.split('[，。！？、‘’“”]', text)
        labels = [[self.idx2tag[idx] for idx in seq] for seq in paths]

        res = []
        for sent, tags in zip(seqs, labels):
            tags = self._correct_tags(tags)
            res.append(list(zip(sent, tags)))
        return res
    
    def _correct_tags(self, tags):
        stack = []
        for idx, tag in enumerate(tags):
            if tag.startswith('B'):
                stack.append(tag)
            elif tag.startswith('M') and stack and tags[stack[-1]] == 'B_' + tag[2:]:
                continue
            elif tag.startswith('E') and stack and tags[stack[-1]] == 'B_' + tag[2:]:
                stack.pop()
            else:
                stack.append(idx)

        for idx in stack:
            tags[idx] = 'O'

        return tags


origin_handle_entities()
origin_handle_mark()
sentence_split()
data_to_pkl()
load_data_rm()


token2idx, idx2token, tag2idx, idx2tag = read_Word2vec()
N = len(tag2idx)
M = len(token2idx)
pi, A, B = save_hmm()

text = '寒武纪在北京举办2022春季专场招聘会'
vocab_file = 'vocab.pkl'
tags_file = 'tags.pkl'
tokenizer = Tokenizer(vocab_file)
input_ids = tokenizer.encode(text, maxlen=40)
parser = Parser(tags_file)
paths = predict(input_ids)
print(parser.decode(text, paths))

