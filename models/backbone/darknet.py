from typing import List, Tuple, Any, Optional

import torch
import torch.nn as nn

from ..block import ConvBNAct, C3k2, SPPF, C2PSA, UniversalStem


def build_layer(
    in_channels,
    out_channels,
    num_blocks,
    block_args,
    downsample = True,
    dim_increase = False
):
    """构建 Darknet 的层，包含下采样 ConvBNAct + C3k2 块堆叠"""
    layers = []
    c3k = block_args[0]
    expansion = block_args[1] if len(block_args) > 1 else 0.5
    if downsample:
        layers.append(
            ConvBNAct(
                in_channels = in_channels,
                out_channels = in_channels if not dim_increase else out_channels,
                kernel_size = 3,
                stride = 2))
    layers.append(
        C3k2(
            in_channels = in_channels if not dim_increase else out_channels,
            out_channels = out_channels,
            num_blocks = num_blocks,
            c3k = c3k,
            expansion = expansion))
    return nn.Sequential(*layers)


class Darknet(nn.Module):
    """
    Darknet 模型
    示例图：
    """
    def __init__(
        self,
        *,
        layer_channels: Optional[List[int]] = [128, 256, 512, 512, 512],
        num_blocks: Optional[List[int]] = [2, 2, 2, 2, 2],
        block_args: List[Tuple[Any]] = [(True, .25), (True, .25), (True,), (True,)],
        in_channels: int = 3
    ):
        super().__init__()
        self.layer_channels = layer_channels
        self.num_blocks = num_blocks
        self.block_args = block_args
        self.out_channels = self.layer_channels[-3:]
        self.stem = nn.Sequential(
            ConvBNAct(
                in_channels = in_channels,
                out_channels = self.layer_channels[0] // 2,
                kernel_size = 3,
                stride = 2),
            ConvBNAct(
                in_channels = self.layer_channels[0] // 2,
                out_channels = self.layer_channels[0],
                kernel_size = 3,
                stride = 2))
        self.layer1 = build_layer(
            in_channels = self.layer_channels[0],
            out_channels = self.layer_channels[1],
            num_blocks = self.num_blocks[0],
            block_args = self.block_args[0],
            downsample = False)
        self.layer2 = build_layer(
            in_channels = self.layer_channels[1],
            out_channels = self.layer_channels[2],
            num_blocks = self.num_blocks[1],
            block_args = self.block_args[1])
        self.layer3 = build_layer(
            in_channels = self.layer_channels[2],
            out_channels = self.layer_channels[3],
            num_blocks = self.num_blocks[2],
            block_args = self.block_args[2])
        self.layer4 = build_layer(
            in_channels = self.layer_channels[3],
            out_channels = self.layer_channels[4],
            num_blocks = self.num_blocks[3],
            block_args = self.block_args[3],
            dim_increase = True)
        self.sppf = SPPF(in_channels=self.layer_channels[4], out_channels=self.layer_channels[4])
        self.c2psa = C2PSA(
            in_channels = self.layer_channels[4],
            out_channels = self.layer_channels[4],
            num_blocks = self.num_blocks[4])

    def forward(self, x):
        x_4 = self.stem(x)
        x_4 = self.layer1(x_4)
        x_8 = self.layer2(x_4)
        x_16 = self.layer3(x_8)
        x_32 = self.layer4(x_16)
        x_32 = self.sppf(x_32)
        x_32 = self.c2psa(x_32)

        return x_8, x_16, x_32