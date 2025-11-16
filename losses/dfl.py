import torch
import torch.nn as nn
import torch.nn.functional as F


def dfl_loss(pred_dist, target, reg_max=15):
    """
    Distribution Focal Loss (DFL)。
    Args:
        pred_dist: 预测的分布 logits，形状 [N, 4*(reg_max+1)]
        target: 真实的 bbox 偏移（已归一化到 [0, reg_max]），形状 [N, 4]
        reg_max: 最大回归值，默认 15（对应 16 个离散点）
    
    返回:
        loss: 标量，平均 DFL 损失
    """
    # 将 target 映射到 [0, reg_max] 区间
    target = target.clamp(min=0, max=reg_max)
    # reshape pred_dist: [N, 4*(reg_max+1)] -> [4*N, reg_max+1]
    batch_size = pred_dist.size(0)
    pred_dist = pred_dist.view(-1, reg_max + 1)  # [4*N, reg_max+1]
    # 构造 soft label（两相邻整数之间的插值）
    # 例如 target=9.7 → label 在 index 9 和 10 上分配权重 0.3 和 0.7
    target = target.view(-1)  # [4*N]
    left = target.long()      # floor
    right = left + 1
    weight_left = (right - target).unsqueeze(-1)   # [4*N, 1]
    weight_right = (target - left).unsqueeze(-1)   # [4*N, 1]
    # 构造类似软标签的 one-hot
    target_dist = torch.zeros_like(pred_dist)  # [4*N, reg_max+1]
    target_dist.scatter_(1, left.unsqueeze(1), weight_left)
    target_dist.scatter_(1, right.unsqueeze(1), weight_right)
    # 使用 softmax + cross entropy（等价于 KL 散度，因 target_dist 是概率分布）
    # loss = F.cross_entropy(pred_dist, target_dist, reduction='none')
    loss = torch.sum(-target_dist * F.log_softmax(pred_dist, dim=-1), dim=-1)

    return loss.mean()