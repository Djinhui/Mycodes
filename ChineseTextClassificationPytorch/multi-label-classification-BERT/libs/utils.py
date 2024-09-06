# -*- coding: utf-8 -*-
# @Time : 2022/11/16 11:05
# @Author : TuDaCheng
# @File : utils.py
import re
import os
import time
import torch
import json
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD, CLS, SEP = '<UNK>', '<PAD>', '<CLS>', '<SEP>'  # 未知字，padding符号


def preprocessing_text(text):
    """
    文本预处理 去出停用词和非中文其他符号
    :param text:
    :return:
    """

    def remove_1a():
        # 去除标点字母数字
        chinese = '[\u4e00-\u9fa5]+'
        str1 = re.findall(chinese, text)
        text2 = ''.join(str1)
        return text2

    def delete_boring_characters():
        sentence = remove_1a()
        return re.sub(r'[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", sentence)

    content = delete_boring_characters()
    return content


def build_dataset(config):
    """
    构建数据集
    :param config:
    :return: rain dev test
    """
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, "rb"))
        train = dataset["train"]
        dev = dataset["dev"]
        test = dataset["test"]
    else:
        # 读取标签
        label_list = pkl.load(open(config.label_path, 'rb'))
        print(f"标签个数======== {len(label_list)}")

        def convert_to_one_hot(Y, C):
            list_ = [[0 for i in C] for j in Y]
            for i, a in enumerate(Y):
                if isinstance(a, list):
                    for b in a:
                        if b in C:
                            list_[i][C.index(b)] = 1
                        else:
                            list_[i][len(C) - 1] = 1
                else:
                    continue
            return list_

        def load_dataset(file_path, config):
            """
              返回4个列表： ids, label, seq_length, mask
              :param file_path:
              :param config:
              :return:
              """

            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            all_sentences = []  # 存放所有的文本
            all_labels = []  # 存放所有的标签
            need_labels = config.class_list
            # 提取标签
            all_data = raw_data["data"]
            for one_data in all_data:
                data_lists = one_data["conversation"]
                speaker_list = []
                text_list = []
                for i, dl in enumerate(data_lists):
                    speaker = dl["speaker"]
                    text = dl["text"]
                    labels = dl["labels"]

                    replace_label = ["基本信息-与债务人关系-父母", "基本信息-与债务人关系-配偶", "基本信息-与债务人关系-子女"]
                    for r_label in replace_label:
                        if r_label in labels:
                            labels.append("基本信息-与债务人关系-认识")
                    l = []
                    for label in labels:
                        if label in need_labels:
                            l.append(label)  # 提出不在需要的标签列表中

                    speaker_list.append(speaker)
                    text_list.append(text)

                    if speaker == 2 and i != 0:
                        text = text_list[i - 1] + SEP + text_list[i]
                        all_sentences.append(text)
                        if l:
                            all_labels.append(l)
                        else:
                            all_labels.append("null")
                    elif speaker == 2 and i == 0:
                        text = "" + SEP + text_list[i]
                        all_sentences.append(text)
                        if l:
                            all_labels.append(l)
                        else:
                            all_labels.append("null")

            print(all_labels)

            # 把数组转成独热
            labels_id = convert_to_one_hot(all_labels, label_list)
            contents = []

            for i, text in enumerate(all_sentences):
                texts = text.split(SEP)

                # encoded_dict = config.tokenizer.encode_plus(
                #     texts,  # 输入文本
                #     add_special_tokens=True,  # 添加 '[CLS]' 和 '[SEP]'
                #     max_length=config.pad_size,  # 填充 & 截断长度
                #     pad_to_max_length=True,
                #     padding='max_length',
                #     truncation='only_first',
                #     return_attention_mask=True,  # 返回 attn. masks.
                #     return_tensors='pt'  # 返回 pytorch tensors 格式的数据
                # )
                #
                # label = labels_id[i]
                # token_ids = encoded_dict["input_ids"]
                # mask = encoded_dict["attention_mask"]
                # segment_ids = encoded_dict["token_type_ids"]
                # seq_length = config.pad_size

                content1 = preprocessing_text(texts[0])
                content2 = preprocessing_text(texts[1])

                tokens1 = config.tokenizer.tokenize(content1)
                tokens1 = [CLS] + tokens1 + [SEP]
                tokens1_id = config.tokenizer.convert_tokens_to_ids(tokens1)
                tokens2 = config.tokenizer.tokenize(content2)
                tokens2 = tokens2 + [SEP]
                tokens2_id = config.tokenizer.convert_tokens_to_ids(tokens2)

                token_ids = tokens1_id + tokens2_id
                segment_ids = [1] * len(tokens1)
                segment_padding = [0] * (config.pad_size - len(tokens1))
                segment_ids += segment_padding
                seq_length = len(token_ids)
                label = labels_id[i]

                pad_size = config.pad_size
                if len(token_ids) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
                    token_ids = token_ids + ([0] * (pad_size - len(token_ids)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_length = pad_size
                    segment_ids = segment_ids[:pad_size]

                contents.append((token_ids, label, seq_length, mask, segment_ids))

            return contents

        train = load_dataset(config.train_path, config)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset["train"] = train
        dataset["dev"] = dev
        dataset["test"] = test
        pkl.dump(dataset, open(config.datasetpkl, "wb"))

    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        segments_id = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask, segments_id), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def log(self, *value, end="\n"):

        current = time.strftime("[%Y-%m-%d %H:%M:%S]")
        s = current
        for v in value:
            s += " " + str(v)
        s += end
        print(s, end="")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(s)
