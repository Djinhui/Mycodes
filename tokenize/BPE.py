from transformers import AutoTokenizer
from collections import defaultdict

# 假定训练的语料(已归一化处理)为4个句子
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# 预切分
# init pre tokenize function
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
pre_tokenize_function = gpt2_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
pre_tokenized_corpus = [pre_tokenize_function(text) for text in corpus]
'''
pre_tokenized_corpus:
[
    [('This', (0, 4)), ('Ġis', (4, 7)), ('Ġthe', (7, 11)), ('ĠHugging', (11, 19)), ('ĠFace', (19, 24)), ('ĠCourse', (24, 31)), ('.', (31, 32))], 
    [('This', (0, 4)), ('Ġchapter', (4, 12)), ('Ġis', (12, 15)), ('Ġabout', (15, 21)), ('Ġtokenization', (21, 34)), ('.', (34, 35))], 
    [('This', (0, 4)), ('Ġsection', (4, 12)), ('Ġshows', (12, 18)), ('Ġseveral', (18, 26)), ('Ġtokenizer', (26, 36)), ('Ġalgorithms', (36, 47)), ('.', (47, 48))], 
    [('Hopefully', (0, 9)), (',', (9, 10)), ('Ġyou', (10, 14)), ('Ġwill', (14, 19)), ('Ġbe', (19, 22)), ('Ġable', (22, 27)), ('Ġto', (27, 30)), ('Ġunderstand', (30, 41)), ('Ġhow', (41, 45)), ('Ġthey', (45, 50)), ('Ġare', (50, 54)), ('Ġtrained', (54, 62)), ('Ġand', (62, 66)), ('Ġgenerate', (66, 75)), ('Ġtokens', (75, 82)), ('.', (82, 83))]
]

'''
# 统计每个整词的词频
word2count = defaultdict(int)
for split_text in pre_tokenized_corpus:
    for word, _ in split_text:
        word2count[word] += 1
'''
word2count:
 {'This': 3, 'Ġis': 2, 'Ġthe': 1, 'ĠHugging': 1,...}
'''

# 字符级别的小词表
vocab_set = set()
for word in word2count:
    vocab_set.update(list(word))
vocabs = list(vocab_set) 
'''
vocabs:
['i', 't', 'p', 'o', 'r', 'm', 'e', ',', 'y', 'v', 'Ġ', 'F', 'a', 'C', 'H', '.', 'f', 'l', 'u', 'c', 'T', 'k', 'h', 'z', 'd', 'g', 'w', 'n', 's', 'b']
'''

# 基于小词表对每个整词进行切分
word2splits = {word:[c for c in word] for word in word2count}
'''
word2splits:
'This': ['T', 'h', 'i', 's'], 
'Ġis': ['Ġ', 'i', 's'],
.......

'''

# 基于word2split统计vocabs中相邻两个pair的词频pair2count
def _compute_pair2score(word2splits, word2count):
    pair2count = defaultdict(int)
    for word, word_count in word2count.items():
        split = word2splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split)-1):
            pair = (split[i], split[i+1])
            pair2count[pair] += word_count
    return pair2count
'''
pair2count:
{('T', 'h'): 3, ('h', 'i'): 3, ('i', 's'): 5,...}
'''

# 统计当前频率最高的相邻pair
def _compute_most_score_pair(pair2count):
    best_pair = None
    max_freq = None
    for pair, freq in pair2count.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    return best_pair

# 重新对word2count进行切分
def _merge_pair(a, b, word2splits):
    new_word2splits = dict()
    for word, split in word2splits.items():
        if len(split) == 1:
            new_word2splits[word] = split
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a+b] + split[i+2:]
            else:
                i += 1

        new_word2splits[word] = split
    return new_word2splits


# 训练阶段：给定语料，生成合并规则和词表
merge_rules = []
vocab_size = 50
while len(vocabs) < vocab_size:
    pair2score = _compute_pair2score(word2splits, word2count)
    best_pair = _compute_most_score_pair(pair2score)
    # 更新vocab
    vocabs.append(best_pair[0]+best_pair[1])
    # 把当前best_pair('Ġ', 't')合并成一个词并添加到词表中。同时在合并规则中添加best_pair('Ġ', 't')这条合并规则
    merge_rules.append(best_pair)

    word2splits = _merge_pair(best_pair[0], best_pair[1], word2splits)

# 推理阶段：给定一个句子，将其切分为token序列
def tokenize(text):
    # pre tokenize
    words = [word for word, _ in pre_tokenize_function(text)]
    # split into char level
    splits = [[c for c in word] for word in words]
    # apply merge rules
    for merge_rule in merge_rules:
        for index, split in enumerate(splits):
            i = 0
            while i < len(split)-1:
                if split[i]==merge_rule[0] and split[i+1]==merge_rule[1]:
                    split = split[:i] + ["".join(merge_rule)] + split[i+2:]
                else:
                    i+=1
            splits[index] = split
    return sum(splits,[])
    