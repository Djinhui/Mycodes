# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# 编写公式中计算分数的函数, 依次进行log, 求和, 指数e计算
def log_sum_exp(vec, m_size):
    # vec: size=(batch_size, vanishing_dim, hidden_dim)
    # m_size: hidden_dim
    # size=(batch_size, hidden_dim)

    # 在第1个维度上求最大值, 贪心算法
    # vec: torch.Size([32, 12, 12])
    _, idx = torch.max(vec, 1)
    # idx: torch.Size([32, 12])
    # 按照idx的下标进行分数取值, 并调整张量形状
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)
    # max_score: torch.Size([32, 1, 12])

    # 此处max_score外提到最前面, 是一种数学上防止浮点数溢出的技巧, 内部则依次进行log, sum, exp的计算
    return max_score.view(-1, m_size) + torch.log(torch.sum(
        torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)


# 完成CRF核心类的构建
class CRF(nn.Module):
    def __init__(self, **kwargs):
        super(CRF, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        
        # 初始化赋值, 将<start>, <end>人为标注成-2, -1
        self.START_TAG_IDX, self.END_TAG_IDX = -2, -1
        init_transitions = torch.zeros(self.target_size + 2, self.target_size + 2)
       
        # 任意合法的转移, 都不能转移到<start>节点; 任意合法的转移, 都不能从<eos>节点转移出来
        init_transitions[:, self.START_TAG_IDX] = -1000.
        init_transitions[self.END_TAG_IDX, :] = -1000.
        
        if self.use_cuda:
            init_transitions = init_transitions.cuda()
        # 将转移矩阵设置成nn.Parameter(), 即成为模型参数的一部分参与反向传播, 并更新参数.
        self.transitions = nn.Parameter(init_transitions)

    def _forward_alg(self, feats, mask=None):
        # feats: (batch_size, seq_len, self.target_size + 2)
        # mask:  (batch_size, seq_len)
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        # 设置mask等参数
        mask = mask.transpose(1, 0).contiguous()
        ins_num = batch_size * seq_len
        # mask: torch.Size([256, 32])
        # feats: torch.Size([32, 256, 12])
        # 将feats张量进行shape的调整
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # feats: torch.Size([8192, 12, 12])

        # 前向传播张量feats, 加上转移概率值, 得到前向传播分数
        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        # scores: torch.Size([8192, 12, 12])
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # scores: torch.Size([256, 32, 12, 12])
        seq_iter = enumerate(scores)

        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()

        # 得到起始标签的初始张量
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        # partition: torch.Size([32, 12, 1])

        # 按照动态规划的算法, 遍历所有可能的转移, 计算最优的路径分数
        for idx, cur_values in seq_iter:
            # cur_values: torch.Size([32, 12, 12])
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # cur_values: torch.Size([32, 12, 12])
            cur_partition = log_sum_exp(cur_values, tag_size)
            # cur_partition: torch.Size([32, 12])
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            # mask_idx: torch.Size([32, 12])
            masked_cur_partition = cur_partition.masked_select(mask_idx.bool())
            # masked_cur_partition: torch.Size([384])

            # 进行掩码张量的遮掩计算
            if masked_cur_partition.dim() != 0:
                mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
                # mask_idx: torch.Size([32, 12, 1])
                # partition: torch.Size([32, 12, 1])
                partition.masked_scatter_(mask_idx.bool(), masked_cur_partition)
                # partition: torch.Size([32, 12, 1])

        # 状态转移矩阵 + START起始状态矩阵
        cur_values = self.transitions.view(1, tag_size, tag_size).expand(
                          batch_size, tag_size, tag_size) + partition.contiguous().view(
                          batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        # cur_values: torch.Size([32, 12, 12])
        # 进行log, sum, exp的公式计算
        cur_partition = log_sum_exp(cur_values, tag_size)
        # cur_partition: torch.Size([32, 12])
        # 整个转移矩阵达到终点END
        final_partition = cur_partition[:, self.END_TAG_IDX]
        # final_partition: torch.Size([32])
        # tensor([ 46.8448,  81.1508, 198.9986,  34.6561,  83.6654, 107.4941,  66.8414,
        #          116.7521,  44.4528,  99.5374,  58.1670,  89.1694,  90.9867,  44.5918,
        #          144.7855,  58.3682,  43.8786,  88.5129, 133.5388,  37.9694, 195.4293,
        #          155.1181,  28.5585, 206.8521,  94.5196,  69.2064,  76.8546, 182.8614,
        #          88.4160, 196.1000,  83.3181,  78.3477], device='cuda:0',
        #          grad_fn=<SelectBackward>)
        # scores: torch.Size([256, 32, 12, 12])

        # 返回当前batch分数的总和, 还有scores张量
        return final_partition.sum(), scores

    # 维特比解码核心函数
    def _viterbi_decode(self, feats, mask=None):
        # feats: (batch_size, seq_len, self.target_size+2)
        # mask:  (batch_size, seq_len)
        # decode_idx: (batch_size, seq_len)   --- viterbi decode结果
        # path_score: (batch_size, 1)         --- 每个句子的得分
        # 获取关键参数值
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(-1)

        # mask: torch.Size([32, 256])
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        mask = mask.transpose(1, 0).contiguous()
        # mask: torch.Size([256, 32])
        # length_mask: torch.Size([32, 1])
        ins_num = seq_len * batch_size
        # feats: torch.Size([32, 256, 12])
        # 和前向传播一样的逻辑, 进行feats张量还有转移矩阵的分数计算
        feats = feats.transpose(1, 0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        # feats: torch.Size([8192, 12, 12])

        scores = feats + self.transitions.view(1, tag_size, tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # scores: torch.Size([256, 32, 12, 12])

        seq_iter = enumerate(scores)
        # 记录最优得分的位置信息
        back_points = list()
        partition_history = list()
        mask = (1 - mask.long()).bool()

        try:
            _, inivalues = seq_iter.__next__()
        except:
            _, inivalues = seq_iter.next()
        
        partition = inivalues[:, self.START_TAG_IDX, :].clone().view(batch_size, tag_size, 1)
        # partition: torch.Size([32, 12, 1])
        partition_history.append(partition)

        # 按照动态规划算法, 计算所有状态的分数
        for idx, cur_values in seq_iter:
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # cur_values: torch.Size([32, 12, 12])
            partition, cur_bp = torch.max(cur_values, 1)
            # partition: torch.Size([32, 12])
            # cur_bp: torch.Size([32, 12])
            partition_history.append(partition.unsqueeze(-1))

            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0)
            # cur_bp: torch.Size([32, 12])
            back_points.append(cur_bp)

        # 集合所有位置的历史信息
        partition_history = torch.cat(partition_history).view(seq_len, batch_size, -1).transpose(1, 0).contiguous()
        # partition_history: torch.Size([32, 256, 12])

        # 再次进行掩码张量的计算和遮掩
        last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
        # last_position: torch.Size([32, 1, 12])
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size, 1)
        # last_position: torch.Size([32, 1, 12])

        # 将最后的分数矩阵调整成方阵
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + \
            self.transitions.view(1, tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        # last_values: torch.Size([32, 12, 12])

        # 在第1个维度上按照贪心解码计算最优解
        _, last_bp = torch.max(last_values, 1)
        # last_bp: torch.Size([32, 12])
        
        # 增加全零张量, 方便进行反向遍历
        pad_zero = Variable(torch.zeros(batch_size, tag_size)).long()
        if self.use_cuda:
            pad_zero = pad_zero.cuda()
        
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)
        # back_points: torch.Size([256, 32, 12])

        # pointer指向END节点, 方便后续的反向回溯
        pointer = last_bp[:, self.END_TAG_IDX]
        # pointer: torch.Size([32])
        insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
        # insert_last: torch.Size([32, 1, 12])
        back_points = back_points.transpose(1, 0).contiguous()
        # back_points: torch.Size([32, 256, 12])

        back_points.scatter_(1, last_position, insert_last)
        # back_points: torch.Size([32, 256, 12])

        back_points = back_points.transpose(1, 0).contiguous()
        # back_points: torch.Size([256, 32, 12])

        decode_idx = Variable(torch.LongTensor(seq_len, batch_size))
        
        if self.use_cuda:
            decode_idx = decode_idx.cuda()
        
        # 进行反向回溯的解码
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            # pointer: torch.Size([32, 1])
            decode_idx[idx] = pointer.view(-1).data
        
        path_score = None
        decode_idx = decode_idx.transpose(1, 0)
        # decode_idx: torch.Size([32, 256])
        # tensor([[0, 7, 2,  ..., 0, 0, 2],
        #         [2, 0, 2,  ..., 0, 0, 9],
        #         [3, 2, 1,  ..., 0, 0, 9],
        #         ...,
        #         [4, 9, 0,  ..., 0, 0, 5],
        #         [2, 0, 0,  ..., 0, 0, 5],
        #         [2, 0, 2,  ..., 0, 0, 9]], device='cuda:0')
        return path_score, decode_idx

    # 在进行inference的时候所调用的forward函数, 仅仅在发射张量上进行维特比解码
    def forward(self, feats, mask=None):
        path_score, best_path = self._viterbi_decode(feats, mask)
        return path_score, best_path

    # 文本语句的原始分数计算
    def _score_sentence(self, scores, mask, tags):
        # scores: (seq_len, batch_size, tag_size, tag_size)
        # mask:   (batch_size, seq_len)
        # tags:   (batch_size, seq_len)
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(-1)

        new_tags = Variable(torch.LongTensor(batch_size, seq_len))
        if self.use_cuda:
            new_tags = new_tags.cuda()
        
        for idx in range(seq_len):
            if idx == 0:
                new_tags[:, 0] = (tag_size - 2) * tag_size + tags[:, 0]
            else:
                new_tags[:, idx] = tags[:, idx - 1] * tag_size + tags[:, idx]

        # new_tags: torch.Size([32, 256])
        # tensor([[128,  96,   0,  ...,  91,  91,  91],
        #         [128,  96,   0,  ...,  91,  91,  91],
        #         [128,  96,   0,  ...,  91,  91,  91],
        #         ...,
        #         [128,  96,   0,  ...,  91,  91,  91],
        #         [128,  96,   0,  ...,  91,  91,  91],
        #         [128,  96,   0,  ...,  91,  91,  91]], device='cuda:0')
        end_transition = self.transitions[:, self.END_TAG_IDX].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        length_mask = torch.sum(mask, dim=1).view(batch_size, 1).long()
        end_ids = torch.gather(tags, 1, length_mask - 1)
        # end_ids: torch.Size([32, 1])
        # tensor([[9],
        #         [9],
        #         [9],
        #         ...
        #         [9],
        #         [9],
        #         [9]], device='cuda:0')

        end_energy = torch.gather(end_transition, 1, end_ids)
        # end_energy: torch.Size([32, 1])
        # tensor([[0.],
        #         [0.],
        #         [0.],
        #         ...,
        #         [0.],
        #         [0.],
        #         [0.]], device='cuda:0', grad_fn=<GatherBackward>)

        new_tags = new_tags.transpose(1, 0).contiguous().view(seq_len, batch_size, 1)
        # new_tags: torch.Size([256, 32, 1])
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)

        # tg_energy: torch.Size([256, 32])
        tg_energy = tg_energy.masked_select(mask.transpose(1, 0))
        # tg_energy: torch.Size([1480])

        gold_score = tg_energy.sum() + end_energy.sum()
        # tensor(-72.7821, device='cuda:0', grad_fn=<AddBackward0>)

        return gold_score

    # 计算最大似然损失
    def neg_log_likelihood_loss(self, feats, mask, tags):
        # feats: (batch_size, seq_len, tag_size)
        # mask:  (batch_size, seq_len)
        # tags:  (batch_size, seq_len)
        batch_size = feats.size(0)
        mask = mask.bool()

        # 1: 计算前向传播的分数
        forward_score, scores = self._forward_alg(feats, mask)
        # forward_score: torch.Size([])
        # scores: torch.Size([256, 32, 12, 12])
        # 2: 计算文本真实分数
        gold_score = self._score_sentence(scores, mask, tags)
        # gold_score: torch.Size([])

        # 返回batch数据的平均损失值
        if self.average_batch:
            return (forward_score - gold_score) / batch_size
        
        # 3: 返回两者之差, 作为最大似然损失值
        return forward_score - gold_score

