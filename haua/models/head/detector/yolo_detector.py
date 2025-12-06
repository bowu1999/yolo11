from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...block import ConvBNAct


class DualBranchHead(nn.Module):
    """
    图示双分支模块（输入并行到两路）:
      - 回归路: ConvBNAct(3) -> ConvBNAct(3) -> Conv(1) -> out
      - 分类路: ConvBNAct(3) -> ConvBNAct(1) -> ConvBNAct(3) -> ConvBNAct(1) -> Conv(1) -> out
    最终两路的输出 concat 合并
    """
    def __init__(
        self,
        in_channels: int,
        categories: int,
        box_channels: int,
        locate_hidden_channels: int,
        classify_hidden_channels: int,
    ):
        super().__init__()
        self.locate = nn.Sequential(
            ConvBNAct(in_channels=in_channels, out_channels=locate_hidden_channels, kernel_size=3),
            ConvBNAct(
                in_channels = locate_hidden_channels,
                out_channels = locate_hidden_channels,
                kernel_size = 3),
            nn.Conv2d(
                in_channels = locate_hidden_channels,
                out_channels = box_channels,
                kernel_size = 1))
        self.classify = nn.Sequential(
            ConvBNAct(
                in_channels = in_channels,
                out_channels = in_channels,
                kernel_size = 3,
                groups = in_channels),
            ConvBNAct(in_channels=in_channels, out_channels=classify_hidden_channels, kernel_size=1),
            ConvBNAct(
                in_channels = classify_hidden_channels,
                out_channels = classify_hidden_channels,
                kernel_size = 3,
                groups = classify_hidden_channels),
            ConvBNAct(
                in_channels = classify_hidden_channels,
                out_channels = classify_hidden_channels,
                kernel_size = 1),
            nn.Conv2d(in_channels=classify_hidden_channels, out_channels=categories, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.locate(x), self.classify(x)), 1)


class YOLODetector(nn.Module):
    def __init__(
        self,
        in_channels_list: Tuple[int, int, int],
        num_classes: int,
        locate_hidden_channels: Union[Tuple[int, int, int], List[int], int],
        classify_hidden_channels: Union[Tuple[int, int, int], List[int], int],
    ):
        super().__init__()
        assert len(in_channels_list) == 3, "in_channels_list 必须是三个正整数"
        self.in_channels_list = tuple(in_channels_list)
        self.num_classes = num_classes
        self.share_block = len(set(in_channels_list)) == 1
        self.locate_hidden_channels = locate_hidden_channels \
            if isinstance(locate_hidden_channels, (tuple, list)) \
                else (locate_hidden_channels,) * 3
        self.classify_hidden_channels = classify_hidden_channels \
            if isinstance(classify_hidden_channels, (tuple, list)) \
                else (classify_hidden_channels,) * 3
        heads = []
        if self.share_block:
            shared_head = DualBranchHead(
                in_channels = self.in_channels_list[0],
                categories = self.num_classes,
                box_channels = 64,
                locate_hidden_channels = self.locate_hidden_channels[0],
                classify_hidden_channels = self.classify_hidden_channels[0])
            self.heads = nn.ModuleList([shared_head, shared_head, shared_head])
        else:
            module_list = []
            for inx, in_ch in enumerate(self.in_channels_list):
                h = DualBranchHead(
                    in_channels = self.in_channels_list[inx],
                    categories = self.num_classes,
                    box_channels = 64,
                    locate_hidden_channels = self.locate_hidden_channels[inx],
                    classify_hidden_channels = self.classify_hidden_channels[inx])
                module_list.append(h)
            self.heads = nn.ModuleList(module_list)

    def forward(
        self,
        features: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(features, (list, tuple)) and len(features) == 3, \
            "features must be a list/tuple of 3 tensors"
        outs = tuple()
        for i, x in enumerate(features):
            out = self.heads[i](x)
            outs += (out,)

        return outs
