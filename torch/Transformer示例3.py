# https://www.zhihu.com/question/362131975/answer/2182682685
# 下面所有的实现代码都是笔者直接从Pytorch 1.4版本中torch.nn.Transformer模块里摘取出来的简略版
# see《trm_nlp案例》

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.nn.init import xavier_uniform_

def multi_head_attention_forward(
        query, # [tgt_len, batch_size, embed_dim]
        key, # [src_len, batch_size, embed_dim]
        value, # [src_len, batch_size, embed_dim]
        num_heads, dropout_p, 
        out_proj_weight, # [embed_dim=vdim*num_heads, embed_dim]
        out_proj_bias,
        training=True,
        key_padding_mask=None, #[batch_size, src_len(tgt_len)]
        q_proj_weight = None, # [embed_dim, kdim*num_heads]
        k_proj_weight = None, # [embed_dim, kdim*num_heads]
        v_proj_weight = None, # [embed_dim, vdim*num_heads]
        attn_mask = None, # [tgt_len, src_len]
        ):
    q = F.linear(query, q_proj_weight) # [tgt_len, batch_size, embed_dim]*[embed_dim, kdim*num_heads]=[tgt_len, batch_size, kdim*num_heads]
    k = F.linear(key, k_proj_weight) # [src_len, batch_size, embed_dim]*[embed_dim, kdim*num_heads]=[src_len, batch_size,kdim*num_heads]
    v = F.linear(value, v_proj_weight) # [src_len, batch_size, embed_dim]*[embed_dim, vdim*num_heads]=[src_len, batch_size,vdim*num_heads]

    tgt_len, bsz, embed_dim = query.size()
    src_len = key.size(0)
    head_dim = embed_dim // num_heads
    scaling = float(head_dim) ** -0.5
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2: # [tgt_len, src_len]
            attn_mask = attn_mask.unsqueeze(0) # [1, tgt_len, src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3: # [num_heads*batch_sizem, tgt_len, src_len]
            if list(attn_mask.size) != [bsz*num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of 3D attn_mask is not correct')
            
    q = q.contiguous().view(tgt_len, bsz*num_heads, head_dim).transpose(0, 1) # [batch_size*num+heads, tgt_len, kdim]
    k = k.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0, 1) # [batch_size*num+heads, src_len, kdim]
    v = v.contiguous().view(-1, bsz*num_heads, head_dim).transpose(0, 1) # [batch_size*num+heads, src_len, vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1,2)) # [batch_size*num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask # [batch_size*num_heads, tgt_len, src_len]
    
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz*num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1) # [batch_size*num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v) # [batch_size*num_heads, tgt_len, vdim]

    attn_output = attn_output.transpose(0,1).contiguous().view(tgt_len, bsz, embed_dim) # embed_dim=num_heads*vdim
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = F.linear(attn_output, out_proj_weight, out_proj_bias) # [tgt_len, bsz, embed_dim]
    return Z, attn_output_weights.sum(dim=1)/num_heads # 将num_heads个注意力权重矩阵按对应维度取平均


class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        """
        :param embed_dim:   词嵌入的维度，也就是前面的d_model参数，论文中的默认值为512
        :param num_heads:   多头注意力机制中多头的数量，也就是前面的nhead参数， 论文默认值为 8
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.dkim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim
        self.q_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.k_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.v_proj_weight = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中编码时query, key, value 都是同一个输入， 
        解码时 输入的部分也都是同一个输入， 
        解码和编码交互时 key,value指的是 memory, query指的是tgt
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                一般只在解码时使用,为了并行一次喂入所有解码部分的输入,所以要用mask来进行掩盖当前时刻之后的位置信息
                tgt_len本质上指的其实是query_len;src_len本质上指的是key_len。只是在不同情况下两者可能会是一样,也可能会是不一样
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads, self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training, key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight, attn_mask=attn_mask)
    

src_len = 5
batch_size = 2
dmodel = 32
num_head = 1
src = torch.rand((src_len, batch_size, dmodel))
src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                     [True, True, True, True, False]])
'''
src = torch.tensor([[76,45,32,0,0],
                    [32,56,78,97,0]])
mask = torch.eq(src, 0)
src_key_padding_mask = torch.tensor([[False, False, False, True, True],
                                     [False, False, False, False, True]])
'''

my_mh = MyMultiheadAttention(embed_dim=dmodel, num_heads=num_head)
r = my_mh(src,src,src, key_padding_mask=src_key_padding_mask)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emd_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emd_size)
        self.emd_size = emd_size

    def forward(self, tokens):
        '''
        :param tokens: shape [len, batch_size]
        :return: shape [len, batch, emd_size]
        '''
        return self.embedding(tokens.long()) * math.sqrt(self.emd_size)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsuqeeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        :param x: shape [x_len, batch_size, emb_size]
        :return : [x_len, batch_size, emd_size]
        '''
        x = x + self.pe[:x.size(0), :] # (x_len, batch_size, d_model)
        return self.dropout(x)
    
x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
x = x.reshape(5, 2)  # [src_len, batch_size]
token_embedding = TokenEmbedding(vocab_size=11, emb_size=512)
x = token_embedding(tokens=x)
pos_embedding = PositionalEncoding(d_model=512)
x = pos_embedding(x=x)
print(x.shape) # torch.Size([5, 2, 512])



class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        '''
        :param src : [src_len, batch_size, embed_dim]
        :param src_mask: [batch_size, src_len]

        '''
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src #  [src_len, batch_size, embed_dim]
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None) -> None:
        super(MyTransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output # [src_len, batch_size, embed_dim]


src_len = 5
batch_size = 2
dmodel = 32
num_head = 3
num_layers = 2
src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                     [True, True, True, True, False]])  # shape: [batch_size, src_len]

my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=dmodel, nhead=num_head)
my_transformer_encoder = MyTransformerEncoder(encoder_layer=my_transformer_encoder_layer,
                                                num_layers=num_layers,
                                                norm=nn.LayerNorm(dmodel))
memory = my_transformer_encoder(src=src, mask=None, 
                                src_key_padding_mask=src_key_padding_mask)
print(memory.shape)  # torch.Size([5, 2, 32])


class MyTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model,nhead, dim_feedforward=2048, dropout=0.1):
        super(MyTransformerDecoderLayer, self).__init__()

        self.self_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head, dropout=dropout) # MaskedSelfAttn
        self.multihead_attn = MyMultiheadAttention(embed_dim=d_model, num_heads=num_head, dropout=dropout) # CrossSelfAttn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param tgt:  解码部分的输入，形状为 [tgt_len,batch_size, embed_dim]
        :param memory: 编码部分的输出memory, [src_len,batch_size,embed_dim]
        :param tgt_mask: 注意力Mask输入,用于掩盖当前position之后的信息, [tgt_len, tgt_len]
        :param memory_mask: 编码器-解码器交互时的注意力掩码,一般为None
        :param tgt_key_padding_mask: 解码部分输入的padding情况,形状为 [batch_size, tgt_len]
        :param memory_key_padding_mask: 编码部分输入的padding情况,形状为 [batch_size, src_len]
        :return:# [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.activation(self.linear1(tgt))
        tgt2 = self.linear2(self.dropout(tgt2))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt # [tgt_len, batch_size, emded_dim]
    
class MyTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super(MyTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len, batch_size, embed_dim]
    

class MyTransformer(nn.Module):
    def __init__(self, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(MyTransformer, self).__init__()
        encoder_layer = MyTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decocer_layer = MyTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = MyTransformerDecoder(decocer_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

        # Final
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        :param src:   [src_len,batch_size,embed_dim]
        :param tgt:  [tgt_len, batch_size, embed_dim]
        :param src_mask:  None 编码时不需要对当前时刻之后的位置信息进行掩盖
        :param tgt_mask:  [tgt_len, tgt_len] 掩盖解码输入中当前时刻以后的所有位置信息
        :param memory_mask: None
        :param src_key_padding_mask: [batch_size, src_len] 对编码输入序列填充部分的Token进行mask
        :param tgt_key_padding_mask: [batch_size, tgt_len] 对解码输入序列填充部分的Token进行掩盖
        :param memory_key_padding_mask:  [batch_size, src_len] 对编码器的输出部分进行掩盖,掩盖原因等同于编码输入时的mask操作
        :return: [tgt_len, batch_size, num_heads * kdim] <==> [tgt_len,batch_size,embed_dim]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask, 
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask) # [tgt_len, batch_size, embed_dim]
        
        output = self.linear(output)
        output = self.softmax(output)

        
        return output # [tgt_len, batch_size, tgt_vocab_size]
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz,sz)) == 1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask
    


src_len = 5
batch_size = 2
dmodel = 32
tgt_len = 6
num_head = 8
src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                            [True, True, True, True, False]])  # shape: [batch_size, src_len]

tgt = torch.rand((tgt_len, batch_size, dmodel))  # shape: [tgt_len, batch_size, embed_dim]
tgt_key_padding_mask = torch.tensor([[True, True, True, False, False, False],
                    [True, True, True, True, False, False]])  # shape: [batch_size, tgt_len]

my_transformer = MyTransformer(d_model=dmodel, nhead=num_head, num_encoder_layers=6,
                                num_decoder_layers=6, dim_feedforward=500)
tgt_mask = my_transformer.generate_square_subsequent_mask(tgt_len)
out = my_transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=src_key_padding_mask)
print(out.shape) #torch.Size([6, 2, 32])