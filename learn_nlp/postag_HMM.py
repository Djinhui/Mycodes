import jieba
import jieba.posseg
# jieba.posseg.cut()
import re
import numpy as np
'''
renmin.txt
/w为标点符号标记
19980101-01-001-001/m  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ——/w  一九九八年/t  新年/t  讲话/n  （/w  附/v  图片/n  １/m  张/q  ）/w  
19980101-02-003-001/m  党中央/nt  国务院/nt  关心/v  西藏/ns  雪灾/n  救灾/vn  工作/vn  
19980101-02-003-002/m  灾区/n  各级/r  政府/n  全力/n  组织/v  抗灾/v  力争/v  降低/v  灾害/n  损失/n  

'''

class HMM:
    def build_transition(self, states_n, state_state_n, states):
        len_status = len(states_n)
        transition_prob = np.zeros((len_status, len_status)) # 状态转移概率矩阵
        for i in range(len_status):
            for j in range(len_status):
                s = states[i] + '_' + states[j]
                tag_i = states[i]
                try:
                    transition_prob[i,j] = state_state_n[s] / (states_n[tag_i]+1)
                except KeyError:
                    transition_prob[i,j] = 0.0
        return transition_prob

    def build_emission(self, states_n, o_state_n, o_sequence, states):
        # 发射概率
        emission_prob = np.zeros((len(states_n), len(o_sequence)))
        for i in range(len(states)):
            for j in range(len(o_sequence)):
                s = o_sequence[j] + '/' + states[i]
                tag_i = states[i]
                try:
                    emission_prob[i,j] = o_state_n[s] / (states_n[tag_i])
                except KeyError:
                    emission_prob[i,j] = 0.0
        return emission_prob

    def viterbi(self, o_sequence, A, B, pi):
        len_status = len(pi)
        status_record = {i:[[0,0] for j in range(o_sequence)] for i in range(len_status)}
        for i in range(len(pi)):
            status_record[i][0][0] = pi[i] * B[i, o_sequence[0]]
            status_record[i][0][1] = 0
        for t in range(1, len(o_sequence)):
            for i in range(len_status):
                max = [-1, 0]
                for j in range(len_status):
                    tmp_prob = status_record[j][t-1][0] * A[j, i]
                    if tmp_prob > max[0]:
                        max[0] = tmp_prob
                        max[1] = j
                status_record[i][t][0] = max[0] * B[i, o_sequence[t]]
                status_record[i][t][1] = max[1]
        return self.get_state_sequence(len_status,o_sequence, status_record)
    
    def get_state_sequence(self, len_status, o_seq, status_record):
        max = 0
        max_idx = 0
        t = len(o_seq) - 1
        for i in range(len_status):
            if status_record[i][t][0] > max:
                max = status_record[i][t][0]
                max_idx = i
        state_sequence = []
        state_sequence.append(max_idx)
        while t > 0:
            max_idx = status_record[max_idx][t][1]
            state_sequence.append(max_idx)
            t -= 1
        return state_sequence[::-1]


class PosTagging:
    def __init__(self):
        self.term_tag_n = {} # 统计单词的次数
        self.tag_tag_n = {} # 词性转移统计
        self.tags_n = {} # 语料库中词性的数量
        self.term_list = [] # 观测序列,单词列表
        self.states = [] # 状态序列,词性列表
        self.hmm = HMM()

    def process_corpus(self, path):
        term_list = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # 处理语料中的前一项时间信息
                line = re.sub("\d{8}-\d{2}-\d{3}-\d{3}/m", "", line)
                sentences = line.split("/w") # /w为标点符号
                """
                ['  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ——',
                 '  一九九八年/t  新年/t  讲话/n  （',
                 '  附/v  图片/n  １/m  张/q  ）',
                 '  \n']
                """
                sentences = [term + '/w' for term in sentences[:-1]] # [-1]为换行符号
                """
                ['  迈向/v  充满/v  希望/n  的/u  新/a  世纪/n  ——/w',
                 '  一九九八年/t  新年/t  讲话/n  （/w',
                 '  附/v  图片/n  １/m  张/q  ）/w']
                """
                for sentence in sentences:
                    terms = sentence.split() #  # ['迈向/v', '充满/v', '希望/n', '的/u', '新/a', '世纪/n', '——/w']
                    for i in range(len(terms)):
                        if terms[i] == '':
                            continue
                        try:
                            self.term_tag_n[terms[i]] += 1
                        except KeyError:
                            self.term_tag_n[terms[i]] = 1
                        word_tag = terms[i].split('/') #['迈向', 'v']
                        term_list.add(word_tag[0])
                        try:
                            self.tags_n[word_tag[-1]] += 1
                        except KeyError:
                            self.tags_n[word_tag[-1]] = 1
                        if i == 0:
                            tag_tag = 'Pos' + '_' + word_tag[-1]
                        else:
                            tag_tag = terms[i-1].split('/')[-1] + '_' + word_tag[-1]
                        try:
                            self.tag_tag_n[tag_tag] += 1
                        except KeyError:
                            self.tag_tag_n[tag_tag] = 1
        self.term_list = list(term_list)
        self.states = list(self.tags_n.keys())

        self.transition = self.hmm.build_transition(self.tags_n, self.tag_tag_n, self.states)
        self.emission = self.hmm.build_emission(self.tags_n, self.term_tag_n, self.term_list, self.states)

        self.build_init_prob()
  
    def build_init_prob(self):
        sum_tag =sum(list(self.tag_tag_n.values()))
        self.pi = [self.tags_n[value]/sum_tag for value in self.tags_n]

    def convert_sentence(self, sentence):
        return [self.term_list.index(term) for term in sentence]


    def predict_tag(self, sentence_cut): # sentence_cut是分词后的数组形式
        o_seq = self.convert_sentence(sentence_cut)
        s_seq = self.hmm.viterbi(o_seq, self.transition, self.emission, self.pi)
        self.out_put_result(o_seq, s_seq, self.term_list, self.states)

    def out_put_result(self, o_seq, s_seq, term_list, states):
        for i in range(len(o_seq)):
            print(term_list[o_seq[i]] + '/' + states[s_seq[i]], end=' ')


    

if __name__ == '__main__':
    pt = PosTagging()
    pt.process_corpus('renmin.txt')
    pt.predict_tag(['你', '可以', '永远', '相信', '这','届','年轻人', '。']) # 你/r 可以/v 永远/d 相信/v 这/r 届/n 年轻人/n 。/w