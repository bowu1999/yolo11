import torch
import torch.nn.functional as F

from .utils import make_grid


def dfl_to_dist(pred_reg: torch.Tensor, reg_max: int = 16, apply_softmax: bool = True):
    """
    将 DFL（Distribution Focal Loss）风格的回归输出转换为连续距离值（l, t, r, b）
    Args:
        pred_reg (Tensor):回归预测值，其中 4 对应 l, t, r, b 四个边距离的分布 logits，形状可为:
                - (B, 4*reg_max, H, W)
                - (B, 4, reg_max, H, W)
        reg_max (int):DFL 的 bins 数（每个边的离散分布长度）。默认 16
        apply_softmax (bool):
            是否对 logits 做 softmax 以获得概率分布，一般为 True，除非你的 logits 已提前 softmax

    Returns:
        dist (Tensor):
            连续的边界偏移量，形状 (B, 4, H, W)，对应 l,t,r,b
            单位仍是“bins 单位”，后续需要乘 stride 得到像素量
    """
    if pred_reg.dim() == 4 and pred_reg.size(1) == 4 * reg_max:
        B, C, H, W = pred_reg.shape
        pred = pred_reg.view(B, 4, reg_max, H, W)
    elif pred_reg.dim() == 5 and pred_reg.size(1) == 4 and pred_reg.size(2) == reg_max:
        pred = pred_reg
    else:
        raise ValueError("预测的形状必须为 (B,4*reg_max,H,W) 或 (B,4,reg_max,H,W)")
    if apply_softmax:
        prob = F.softmax(pred, dim=2)
    else:
        prob = F.softmax(pred, dim=2)
    device = pred.device
    project = torch.arange(reg_max, dtype=prob.dtype, device=device).view(1, 1, reg_max, 1, 1)
    dist = (prob * project).sum(dim=2)  # (B,4,H,W)

    return dist


def decode_dfl(
    pred_reg: torch.Tensor,
    reg_max: int = 16,
    stride: int = 1,
    apply_softmax: bool = True
):
    """
    YOLOv8/YOLO10/YOLO11 风格的 DFL 边界框解码函数
    Args:
        pred_reg (Tensor):边界框回归预测，形状可为：
            - (B, 4*reg_max, H, W)
            - (B, 4, reg_max, H, W)
        reg_max (int):DFL bins 数（每条边的离散概率长度）
        stride (int):当前特征图的 stride，用于将 bins 距离转换为像素单位
        apply_softmax (bool):是否对 logits 应用 softmax。一般为 True

    Returns:
        boxes (Tensor): 解码后的边界框位置，形状 (B, H*W, 4)，格式为 (x1, y1, x2, y2)
        dist_pixels (Tensor): 每个位置的像素级别的偏移量 (l, t, r, b)，形状为 (B,4,H,W)
            可用于 loss 分支等其他用途。
    """
    if pred_reg.dim() == 4:
        _, C, H, W = pred_reg.shape
    else:
        _, _, _, H, W = pred_reg.shape
    dist_bins = dfl_to_dist(pred_reg, reg_max=reg_max, apply_softmax=apply_softmax)  # (B,4,H,W)
    dist_pixels = dist_bins * float(stride)
    grid = make_grid(H, W, stride=stride, device=pred_reg.device, dtype=dist_pixels.dtype)  # (H,W,2)
    cx = grid[..., 0].unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
    cy = grid[..., 1].unsqueeze(0).unsqueeze(1)  # (1,1,H,W)
    l = dist_pixels[:, 0:1, :, :]
    t = dist_pixels[:, 1:2, :, :]
    r = dist_pixels[:, 2:3, :, :]
    b = dist_pixels[:, 3:4, :, :]
    x1 = cx - l
    y1 = cy - t
    x2 = cx + r
    y2 = cy + b
    boxes = torch.cat([x1, y1, x2, y2], dim=1)  # (B,4,H,W)
    boxes = boxes.permute(0, 2, 3, 1).reshape(pred_reg.shape[0], H * W, 4)

    return boxes, dist_pixels
