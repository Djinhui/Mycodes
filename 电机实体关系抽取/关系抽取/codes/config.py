# coding:utf-8
import torch
from fastNLP import Vocabulary
from transformers import BertTokenizer,AdamW
import json



class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_path = '../bert-base-chinese'
        self.num_rel = 253
        self.train_data_path = '../data/train.json'
        self.test_data_path = '../data/test.json'
        self.dev_data_path = '../data/dev.json'
        self.batch_size = 8
        self.rel_dict_path = '../data/rel.json'
        id2rel = json.load(open(self.rel_dict_path,encoding='utf8'))  # 将id与关系名对应，生成一个词汇表对象
        self.rel_vocab = Vocabulary(unknown=None,padding=None)   # 创建一个名为rel_vocab的Vocabulary对象，未知词和填充词都不需要指定
        self.rel_vocab.add_word_lst(list(id2rel.values()))   # 将id2rel字典中的关系名称列表添加到rel_vocab中作为词汇表
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.max_length = 512
        self.learning_rate = 1e-5
        self.bert_dim = 768
        self.epochs = 2
