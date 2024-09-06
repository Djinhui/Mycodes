from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
import torch


# 多标签平衡损失 https://kexue.fm/archives/7359
class MultiLabelBalancedLoss:

    def __call__(self, y_pred, y_true):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
             本文。
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return torch.mean(neg_loss + pos_loss)


# def focalloss(logits, targets, gamma=0.0, weight=None, reduction='mean'):
#     """N samples, C classes
#     logits: [N, C]
#     targets:[N] range [0, C-1]
#     gamma: factor(default 0.0, that is standard cross entropy)
#     weight: [C]
#     """
#     N, C = logits.size(0), logits.size(1)
#     if weight is not None:
#         weight = weight*torch.ones(C).to('cuda')
#         assert len(weight) == C, 'weight length must be equal to classes number'
#         assert weight.dim() == 1, 'weight dim must be 1'
#
#     else:
#         weight = torch.ones(C).to('cuda')
#
#     prob = F.softmax(logits, dim=1)
#     log_prob = F.log_softmax(logits, dim=1)
#     tar_one_hot = F.one_hot(targets, num_classes=C).type(torch.float32)
#     factor= (1 - prob[range(N), targets]) ** gamma
#
#     loss = factor * weight[targets] * (-log_prob * tar_one_hot).sum(dim=1)
#     if reduction == 'mean':
#         loss = loss.sum() / (weight[targets].sum() + 1e-7)
#     elif reduction == 'none':
#         loss = loss
#
#     return loss