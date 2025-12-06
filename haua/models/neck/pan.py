from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..block import ConvBNAct, C3k2


class C3k2PAN(nn.Module):
    def __init__(
        self,
        in_channels: Union[List[int], Tuple[int, int, int]] = (256, 512, 512),
        out_channels: Union[List[int], Tuple[int, int, int]] = (256, 512, 512),
        c3_blocks: int = 1
    ):
        super().__init__()
        assert len(in_channels) == 3, "输入必须是三层特征图 (x_8, x_16, x_32)"
        self.x_8_channels, self.x_16_channels, self.x_32_channels = in_channels
        assert len(out_channels) == 3, "输出必须是三层特征图 (n_8, n_16, n_32)"
        self.out_8_channels, self.out_16_channels, self.out_32_channels = out_channels
        self.top2down_c3k2_f4p_16 = C3k2(
            in_channels = self.x_32_channels + self.x_16_channels,
            out_channels = self.out_16_channels,
            num_blocks = c3_blocks,
            c3k = False)
        self.top2down_c3k2_f4p_8 = C3k2(
            in_channels = self.x_16_channels + self.out_16_channels,
            out_channels = self.out_8_channels,
            num_blocks = c3_blocks,
            c3k = False)
        self.bottom2up_downsample_cbs4p_8 = ConvBNAct(
            in_channels = self.out_8_channels,
            out_channels = self.out_8_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1)
        self.bottom2up_c3k2_f4n_16 = C3k2(
            in_channels = self.out_16_channels + self.out_8_channels,
            out_channels = self.out_16_channels,
            num_blocks = c3_blocks,
            c3k = False)
        self.bottom2up_downsample_cbs4n_16 = ConvBNAct(
            in_channels = self.out_16_channels,
            out_channels = self.out_16_channels,
            kernel_size = 3,
            stride = 2,
            padding = 1)
        self.bottom2up_c3k2_t4n_32 = C3k2(
            in_channels = self.x_32_channels + self.out_16_channels,
            out_channels = self.out_32_channels,
            num_blocks = c3_blocks,
            c3k = True)

    def forward(self, features):
        """
        features: (x_8: 80x80, x_16: 40x40, x_32: 20x20)
        middle: (p_8, p_16, p_32) 
        returns: (n_8, n_16, n_32)
        """
        assert len(features) == 3
        x_8, x_16, x_32 = features
        # ---- top-down ----
        p_32 = x_32
        p_32_upsampled = F.interpolate(x_32, size=x_16.shape[-2:], mode='nearest')
        p_16 = self.top2down_c3k2_f4p_16(torch.cat([p_32_upsampled, x_16], dim=1))
        p_16_upsampled = F.interpolate(p_16, size=x_8.shape[-2:], mode='nearest')
        p_8 = self.top2down_c3k2_f4p_8(torch.cat([p_16_upsampled, x_8], dim=1))
        # ---- bottom-up ----
        n_8 = p_8
        p_8_downsample = self.bottom2up_downsample_cbs4p_8(p_8)
        n_16 = self.bottom2up_c3k2_f4n_16(torch.cat([p_8_downsample, p_16], dim=1))
        n_16_downsample = self.bottom2up_downsample_cbs4n_16(n_16)
        n_32 = self.bottom2up_c3k2_t4n_32(torch.cat([n_16_downsample, p_32], dim=1))

        return n_8, n_16, n_32
