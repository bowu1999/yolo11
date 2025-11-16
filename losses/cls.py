import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def __init__(self, loss_type='bce', num_classes=80):
        super().__init__()
        self.loss_type = loss_type
        self.num_classes = num_classes
        if loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("loss_type must be 'bce' or 'ce'")

    def forward(self, pred, target):
        """
        pred: [N, C] logits
        target: 
            - 若 loss_type='ce': [N] 类别索引
            - 若 loss_type='bce': [N, C] one-hot 或软标签
        """
        if self.loss_type == 'bce' and target.dim() == 1:
            target = torch.zeros(
                pred.size(0),
                self.num_classes,
                device = pred.device).scatter_(1, target.unsqueeze(1), 1.0)

        return self.criterion(pred, target)
