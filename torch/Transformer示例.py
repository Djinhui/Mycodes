# https://blog.csdn.net/qq_44766883/article/details/112008655
# Ref:<Transformer详解以及我的三个疑惑和解答>
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. Scaled Dot-Product Attention
class SacledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(SacledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q,k,v, scale=None, atten_mask=None):
        """
        前向传播 Attention(Q,K,V) = softmax(Q*K.t/sqrt(D_k))*V
        Args:
            q:Queries张量,shape [B, L_q, D_q]
            k:Keys张量,   shape [B, L_k, D_k], D_q=D_k
            v:Values张量, shape [B, L_v, D_v], L_k=L_v
            scale:缩放因子
            atten_mask: Masking张量, shape [B, L_q, L_k]
        
        Returns:
            上下文张量(shape [B, L_q, D_v(即D_q)])和attention张量(shape [B, L_q, L_k])
        """
        attention = torch.bmm(q, k.transpose(1,2)) # out shape [B, L_q, L_k]
        if scale:
            attention = attention * scale
        if atten_mask:
            # 需给mask为True(=1)的地方设置一个负无穷，经过 softmax，这些位置的概率就会接近0
            attention = attention.masked_fill_(atten_mask, -np.inf) 


        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)
        return context, attention


# 2. Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads # 512 // 8 = 64 per head in Paper
        self.num_heads = num_heads
        # 投影矩阵WQ， WK, WV
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = SacledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim) # 拼接投影矩阵
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        # model(inputs, inputs, inputs)
        # model(encoder_out, encoder_out, decoder_in)
        # 前向传播 MultiHeadAttention(Q,K,V) = Concat(softmax(Q*K.t/sqrt(D_k))*V) * W0


        # 残差连接
        residual = query #  for key\value\query: shape (batch_size, L_q, model_dim)
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1) # shape (batch_size * num_heads, L_q, L_k)

        # scaled dot product attention
        scale = (key.size(-1)) ** -0.5 # scale = 1 / sqrt(D_k)

        # self.dot_product_attention 返回形状:
        # context shape (batch_size * num_heads, L_q, dim_per_head)
        # attention shape (batch_size * num_heads, L_q, L_k)
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear project
        output = self.linear_final(context) # shape (batch_size, L_q, model_dim)
        # Add&LN
        output = self.layer_norm(residual + output)
        return output, attention
    
# 3. Mask
'''
- 对于 decoder 的 self-attention里面使用到的 scaled dot-product attention,同时需要padding mask 和 sequence mask 
  作为 attn_mask,具体实现就是两个 mask 相加作为attn_mask

- 其他情况,attn_mask 一律等于 padding mask
'''
def padding_mask(seq_k, seq_q):
    '''padding mask 在所有的 scaled dot-product attention 里面都需要用到'''
    # seq_k和seq_q的shape都是[B,L]
    len_q = seq_q.size(1)
    # 'PAD is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1) # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    '''
    sequence mask 只有在 decoder 的 self-attention 里面用到
    把 时刻t 之后的信息给隐藏起来
    '''
    batch_size, seq_len = seq.size()
    # 上三角的值全为 1，下三角的值全为0，对角线也是0
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1) # [B, L, L]
    return mask


# 4. Positional Embedding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        """
        初始化
        Args:
            d_model: 一个标量。模型的维度,论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
        # 根据论文给的公式，构造出PE矩阵 shape [max_seq_len, d_model]
        position_encoding = np.array([[pos / np.power(10000, 2.0*(j//2) / d_model) for j in range(d_model)] for pos in range(max_seq_len)])
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding  = torch.cat((pad_row, position_encoding))

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len+1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        """
        神经网络的前向传播。

        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor([list(range(1, len_+1)) + [0] * (max_len - len_) for len_ in input_len])
        return self.position_encoding(input_pos)


# 5. Position-wise Feed-Forward network
class PositionalWiseFeedForward(nn.Module):
    '''
    FFN:全连接网络，包含两个线性变换和一个非线性函数
    这里实现上用到了两个一维卷积
    '''
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x): # x (batchsize, seq_len, dim)
        output = x.transpose(1,2) # conv1d need input:(Batch, Channels_in, Length_in)
        output = self.w2(F.relu(self.w1(output))) # conv1d output (Batch, Channels_out, Length_out)
        output = self.dropout(output.transpose(1,2))

        output = self.layer_norm(x + output)
        return output
    

# 6. Encoder
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, atten_mask=None):

        # PART1:self-attention
        context, attention = self.attention(inputs, inputs, inputs, atten_mask)
        # PART2:feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])        

        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=1)     
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)   

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions

# 7. Decoder
class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):

        # PART1: self attention, all inputs are decoder inputs
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # PART2: context attention (cross attention), query is decoder's outputs, key and value are encoder's outputs
        dec_output, context_attention = self.attention(enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # PART3: decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])

        self.seq_embedding = nn.Embedding(vocab_size+1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        self_attn_mask = torch.gt((self_attention_padding_mask+seq_mask), 0)

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_atten, context_attn = decoder(output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_atten)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions
    

# 8. Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len, num_layers=6,
                 model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)

        # Final
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)

        enc_output, enc_self_attn = self.encoder(src_seq, src_len)

        output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, tgt_len, enc_output, context_attn_mask)
        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn
    
    


