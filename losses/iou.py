import torch
import torch.nn as nn


def ciou_loss(pred_boxes, target_boxes):
    """
    计算 CIoU 损失。
    
    参数:   
        pred_boxes: 预测框，形状为 [N, 4]，格式为 (x1, y1, x2, y2)
        target_boxes: 真实框，形状为 [N, 4]，格式为 (x1, y1, x2, y2)
    
    返回:
        ciou_loss: 标量，CIoU 损失值（取平均）
    """
    # 确保输入为浮点型
    pred_boxes = pred_boxes.float()
    target_boxes = target_boxes.float()

    # 计算交集区域的坐标
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # 交集区域的宽和高（若无交集则为0）
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # 各自面积
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # 并集区域
    union_area = pred_area + target_area - inter_area + 1e-7  # 防止除零

    # IoU
    iou = inter_area / union_area

    # 计算最小外接矩形的对角线长度 c^2
    enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclose_c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # 中心点距离 d^2
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    center_d2 = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # 长宽比一致性项
    v = (4 / (torch.pi ** 2)) * torch.pow(
        torch.atan((target_boxes[:, 2] - target_boxes[:, 0]) / (target_boxes[:, 3] - target_boxes[:, 1] + 1e-7)) -
        torch.atan((pred_boxes[:, 2] - pred_boxes[:, 0]) / (pred_boxes[:, 3] - pred_boxes[:, 1] + 1e-7)), 2
    )

    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    # CIoU
    ciou = iou - (center_d2 / enclose_c2 + alpha * v)
    loss = 1.0 - ciou

    return loss.mean()