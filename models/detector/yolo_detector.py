import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

# 假设你已经定义了 Conv（即 CBS: Conv->BN->SiLU）
# class Conv(nn.Module): ... (你在上面给出的实现)

class DualBranchBlock(nn.Module):
    """
    图示双分支模块（输入并行到两路）：
      - 上路：CBS -> CBS -> Conv(1x1) -> out
      - 下路：CBS -> CBS -> CBS -> CBS -> Conv(1x1) -> out
    最终两路的输出可以通过 sum 或 concat 合并（由 merge_mode 决定）。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        merge_mode: str = "sum",  # "sum" or "concat"
        activation: Optional[nn.Module] = None
    ):
        """
        Args:
            in_channels: 输入通道数
            hidden_channels: 分支内部通道（若 None 则取 in_channels）
            out_channels: 分支最后投影到的通道（若 None 则等于 hidden_channels）
            merge_mode: 合并方式 "sum" 或 "concat"
        """
        super().__init__()
        assert merge_mode in ("sum", "concat")
        self.merge_mode = merge_mode

        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = hidden_channels

        # 上路：两个 CBS (Conv) + 1x1 proj
        self.top1 = Conv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.top2 = Conv(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.top_proj = Conv(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        # 下路：四个 CBS + 1x1 proj
        self.bot1 = Conv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.bot2 = Conv(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bot3 = Conv(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bot4 = Conv(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1)
        self.bot_proj = Conv(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)

        # 如果 concat，需要知道合并后通道数
        if self.merge_mode == "concat":
            self._merged_channels = out_channels * 2
        else:
            self._merged_channels = out_channels  # sum -> same channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # top branch
        t = self.top1(x)
        t = self.top2(t)
        t = self.top_proj(t)

        # bottom branch
        b = self.bot1(x)
        b = self.bot2(b)
        b = self.bot3(b)
        b = self.bot4(b)
        b = self.bot_proj(b)

        if self.merge_mode == "sum":
            # 要求 t 和 b 尺寸一致（H,W,C） -> elementwise sum
            return t + b
        else:
            # concat 在通道维度上
            return torch.cat([t, b], dim=1)

    @property
    def merged_channels(self) -> int:
        return self._merged_channels


class Detector(nn.Module):
    """
    多尺度 detector module.
    对每个输入尺度使用一个 DualBranchBlock，然后再用一个最终 1x1 conv 投影到预测维度：
        pred_ch = num_anchors * (5 + num_classes)
    返回: list of prediction tensors（不做 grid decode）
    """
    def __init__(
        self,
        in_channels_list: List[int],
        num_anchors: int,
        num_classes: int,
        hidden_channels: Optional[int] = None,
        merge_mode: str = "sum",
        share_block: bool = False
    ):
        """
        Args:
            in_channels_list: 每个尺度的输入通道列表（例如 [128,256,512]）
            num_anchors: 每尺度锚点数量（常见 3）
            num_classes: 类别数
            hidden_channels: DualBranch 内部隐藏通道（若为 None 则每尺度使用其输入通道）
            merge_mode: DualBranch 合并方式 "sum" or "concat"
            share_block: 是否在所有尺度复用同一个 DualBranchBlock（默认 False：每尺度独立）
        """
        super().__init__()
        assert isinstance(in_channels_list, (list, tuple)) and len(in_channels_list) > 0
        self.num_scales = len(in_channels_list)
        self.pred_ch = num_anchors * (5 + num_classes)

        # 为每个尺度构建 DualBranchBlock
        self.blocks = nn.ModuleList()
        for c in in_channels_list:
            hid = hidden_channels if hidden_channels is not None else c
            block = DualBranchBlock(in_channels=c, hidden_channels=hid, out_channels=hid if merge_mode=="sum" else hid, merge_mode=merge_mode)
            self.blocks.append(block)

        # 若选择共享 block，则把第一个 block 引用赋值给其他
        if share_block and len(self.blocks) > 1:
            for i in range(1, len(self.blocks)):
                self.blocks[i] = self.blocks[0]

        # 最后投影到预测通道：若 merge_mode==concat，merged_channels = hid*2；否则 hid
        self.heads = nn.ModuleList()
        for i, c in enumerate(in_channels_list):
            merged_ch = self.blocks[i].merged_channels
            self.heads.append(Conv(in_channels=merged_ch, out_channels=self.pred_ch, kernel_size=1, stride=1, act=False))

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            feats: list of feature tensors, length == num_scales.
                   对于每个尺度，特征尺寸 (B, C_i, H_i, W_i)
        Returns:
            preds: list of prediction tensors, 每个形状 (B, pred_ch, H_i, W_i)
        """
        assert len(feats) == self.num_scales, "输入特征数量应与 in_channels_list 一致"
        preds = []
        for i, x in enumerate(feats):
            y = self.blocks[i](x)
            p = self.heads[i](y)
            preds.append(p)
        return preds


if __name__ == "__main__":
    # 假设三个尺度：80x80(128ch), 40x40(256ch), 20x20(512ch)
    in_channels = [128, 256, 512]
    model = Detector(in_channels_list=in_channels, num_anchors=3, num_classes=80, hidden_channels=None, merge_mode="sum")
    model.eval()

    b = 2
    f1 = torch.randn(b, 128, 80, 80)
    f2 = torch.randn(b, 256, 40, 40)
    f3 = torch.randn(b, 512, 20, 20)

    outs = model([f1, f2, f3])
    for i, o in enumerate(outs):
        print(f"scale {i} out shape: {o.shape}")
    # 预期输出 (B, 3*(5+80), H, W) -> (2, 255, H, W) 当 num_classes=80, num_anchors=3
