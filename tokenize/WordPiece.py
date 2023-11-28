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
# init pre tojenize function
bert_tokenizer = AutoTokenizer.from_pretrained('bert-bse-cased')
pre_tokenize_function = bert_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
pre_tokenized_corpus = [pre_tokenize_function(text) for text in corpus]

# 统计词频
word2count = defaultdict(int)
for split_text in pre_tokenized_corpus:
    for word, _ in split_text:
        word2count[word] += 1

# 词表
vocab_set = set()
for word in word2count:
    vocab_set.add(word[0])
    vocab_set.update(['##' + c for c in word[1:]]) # 注意这里如果字符不是一个词的开始，需要添加上特殊字符"##"
vocabs = list(vocab_set)

# 基于小词表对每个单词进行切分
word2splits = {word:[word[0]] + ['##' + c for c in word[1:]] for word in word2count}

# 统计相邻pair互信息
def _compute_pair2score(word2splits, word2count):
    vocab2count = defaultdict(int)
    pair2count = defaultdict(int)
    for word, word_count in word2count.items():
        splits = word2splits[word]
        if len(splits) == 1:
            vocab2count[splits[0]] += word_count
            continue
        for i in range(len(splits)-1):
            pair = (splits[i], splits[i+1])
            vocab2count[splits[i]] += word2count
            pair2count[pair] += word2count

        vocab2count[splits[-1]] += word2count

    scores = {pair:freq / (vocab2count[pair[0]]*vocab2count[pair[1]]) for pair, freq in pair2count.items()}
    return scores

# 统计互信息最高的相邻pair
def _compute_most_score_pair(pair2score):
    best_pair = None
    max_score = None
    for pair, score in pair2score.items():
        if max_score is None or max_score < score:
            best_pair = pair
            max_score = score

    return best_pair


def _merge_pair(a, b, word2splits):
    new_pair2splits = dict()
    for word, split in word2splits.items():
        if len(split) == 1:
            new_pair2splits[word] = split
            continue

        i = 0
        while i < len(split)-1:
            if split[i] == a and split[i+1]==b:
                merge = a + b[2:] if b.startswith('##') else a+b
                split = split[:i] + [merge] + split[i+2:]
            else:
                i += 1

        new_pair2splits[word] = split

    return new_pair2splits

vocab_size = 50
while len(vocabs) < vocab_size:
    pair2score = _compute_pair2score(word2count=word2count, word2splits=word2splits)
    best_pair = _compute_most_score_pair(pair2score)
    word2splits = _merge_pair(best_pair[0], best_pair[1],word2splits)

    new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith('##') else best_pair[1]
    vocabs.append(new_token)



# 推理阶段：给定一个句子，将其切分为token序列

def _encode_word(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocabs:
            i -= 1
        if i==0:
            return ['[UNK]']
        
        tokens.append(word[:i])
        word  = word[i:]
        if len(word) > 0:
            word = f'##{word}'
    return tokens

def tokenize(text):
    # pre tokenize
    words = [word for word, _ in pre_tokenize_function(text)]
    encoded_words = [_encode_word(word) for word in words]
    return sum(encoded_words, [])
