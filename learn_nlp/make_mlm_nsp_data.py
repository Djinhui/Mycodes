import re
import math
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# sample IsNext and NotNext to be same in small batch size
def make_batch():
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MLM
        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))
        cand_masked_pos = [i for i, token in enumerate(input_ids) if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]
        shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = word_dict['[MASK]']
            elif random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = word_dict[number_dict[index]]

        n_pad = maxlen - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pred)
            masked_pos.extend([0] * n_pred)
        
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1

    return batch


if __name__ == '__main__':
    maxlen = 30
    batch_size = 6
    max_pred = 5
    n_layers = 6
    n_heads = 12
    d_model = 768
    d_ff = 768*4
    d_k = d_v = 64
    n_segments = 2

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]':0, '[CLS]':1, '[SEP]':2, '[MASK]':3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    number_dict = {i:w for i, w in enumerate(word_dict)}
    vocab_size = len(word_dict)

    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    batch = make_batch()