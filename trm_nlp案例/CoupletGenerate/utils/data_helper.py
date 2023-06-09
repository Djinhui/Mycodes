from collections import Counter
import torch
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

'''
in.txt 上联集合
风 弦 未 拨 心 先 乱 
花 梦 粘 于 春 袖 口 
.
.
.


out.txt 下联集合
夜 幕 已 沉 梦 更 闲 
莺 声 溅 落 柳 枝 头 
.
.
.

'''

def my_tokenizer(s):
    return s.split()

def build_vocab(tokenizer,filepath, min_freq=1, specials=None):
    """
    vocab = Vocab(counter, specials=specials)
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']

    counter = Counter()

    # 上下联共用词表
    with open(filepath[0], encoding='utf-8') as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    with open(filepath[1], encoding='utf-8') as f:
        for string_ in f:
            counter.update(tokenizer(string_))

    return Vocab(counter, specials=specials, min_freq=min_freq)


class LoadCoupletDataset():
    def __init__(self, train_file_paths=None, tokenizer=None, batch_size=32, min_freq=1) -> None:
        # 根据训练预料建立字典，由于都是中文，所以共用一个即可
        self.tokenizer = tokenizer
        self.vocab = build_vocab(self.tokenizer, filepath=train_file_paths, min_freq=min_freq)
        self.specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.PAD_IDX = self.vocab['<pad>']
        self.BOS_IDX = self.vocab['<bos>']
        self.EOS_IDX = self.vocab['<eos>']
        self.batch_size = batch_size

    def data_process(self, filepaths):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        raw_in_iter = iter(open(filepaths[0], encoding="utf8")) # 上联
        raw_out_iter = iter(open(filepaths[1], encoding="utf8")) # 下联
        data = []
        for (raw_in, raw_out) in zip(raw_in_iter, raw_out_iter):
            in_tensor_ = torch.tensor([self.vocab[token] for token in self.tokenizer(raw_in.rstrip('\n'))], dtype=torch.long)
            out_tensor_ = torch.tensor([self.vocab[token] for token in self.tokenizer(raw_out.rstrip('\n'))], dtype=torch.long)
            data.append((in_tensor_, out_tensor_))

        return data
    

    def load_train_val_test_data(self, train_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        test_data = self.data_process(test_file_paths)

        # collate_fn保证每一批次的样本序列长度相同
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, test_iter
    
    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理,该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理,所以这就意味着下面的pad_sequence操作,最终表现出来的结果就是
        不同的样本,padding后在同一个batch中长度是一样的,而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding
        :param data_batch:
        :return:
        """
        in_batch, out_batch = [], []
        for (in_item, out_item) in data_batch:
            in_batch.append(in_item)
            out = torch.cat([torch.tensor(self.BOS_IDX), out_item, torch.tensor([self.EOS_IDX])], dim=0)
            out_batch.append(out)

        in_batch = pad_sequence(in_batch, padding_value=self.PAD_IDX)
        out_batch = pad_sequence(out_batch, padding_value=self.PAD_IDX)
        return in_batch, out_batch
    

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    

    def create_mask(self, src, tgt, device='cpu'):
        src_seq_len = src.size(0)
        tgt_seq_len = tgt.size(0)

        # Decoder的注意力Mask输入，用于掩盖当前position之后的position
        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device) # [tgt_seq_lem, tgt_seq_len]
        # Encoder的注意力Mask输入，这部分其实对于Encoder来说是没有用的，所以这里全是0
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
        # 用于mask掉Encoder的Token序列中的padding部分,[batch_size, src_len]
        src_padding_mask = (src==self.PAD_IDX).transpose(0,1)
        # 用于mask掉Decoder的Token序列中的padding部分,batch_size, tgt_len
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0,1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    







