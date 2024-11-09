# https://blog.csdn.net/weixin_43646592/article/details/130924280#:~:text=RoPE%E5%B0%B1%E6%98%AF%E4%B8%BA%E4%BA%86
# RoPE通过绝对位置编码的方式实现相对位置编码，综合了绝对位置编码和相对位置编码的优点。
# 主要就是对attention中的q, k向量注入了绝对位置信息，然后用更新的q,k向量做attention中的内积就会引入相对位置信息

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)

    # (output_dim//2,)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float) # i, i:[0, d/2]
    theta = torch.pow(10000, -2*ids / output_dim) # θi=10000^(-2i/d)

    # (max_len, output_dim // 2)
    embeddings = position * theta  # pos / (10000^(2i/d))
    
    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (batch_size, nums_head, max_len, output_dm//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

    # (bs, head, max_len, output_dim),reshape后就是：偶数sin，奇数cos
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings


def RoPE(q, k):
    # q, k:(bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[3]

    #(bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)

    # cos_pos, sin_pos:(bs, head, max_len, output_dim)
    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    q2 = torch.stack([-q[...,1::2], q[...,::2]], dim=-1)
    q2 = q2.reshape(q.shape)

    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)

    k = k * cos_pos + k2 * sin_pos

    return q, k


def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q:(bs, head, seq_len, dk)
    # k:(bs, head, seq_len, dk)
    # v:(bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logtis = torch.matmul(q, k.transpose(-2, -1)) # (bs, head, seq_len, seq_len)
    att_logtis /= math.sqrt(d_k)

    if mask is not None:
        att_logtis = att_logtis.masked_fill(mask==0, -1e9)

    att_scores = F.softmax(att_logtis, dim=-1) # (bs, head, seq_len, seq_len)

    if dropout is not None:
        att_scores = dropout(att_scores)

    # (bs, head, seq_len, dk), (bs, head, seq_len, seq_len)
    return torch.matmul(att_scores, v), att_scores


if __name__ == '__main__':
    # (bs, head, seq_len, dk)
    q = torch.randn((8, 12, 10, 32))
    k = torch.randn((8, 12, 10, 32))
    v = torch.randn((8, 12, 10, 32))

    res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)


    # (bs, head, seq_len, dk),  (bs, head, seq_len, seq_len)
    print(res.shape, att_scores.shape)
