from typing import List, Optional, Tuple, Union, Optional

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_padding(
    kernel_size: Union[int, List, Tuple],
    padding: Optional[Union[int, List, Tuple]] = None,
    dilation: int=1
) -> Union[int, Tuple]:
    """
        自动padding，保持输出同输入大小
        Args:
            kernel_size (int | List | Tuple): 卷积核大小
            padding (int | List | Tuple | None): 填充大小，None表示自动计算
            dilation (int): 膨胀率

        Returns:
            padding (int | List | Tuple): 计算后的填充大小
    """
    if dilation > 1:
        kernel_size = dilation * (kernel_size - 1) + 1 \
            if isinstance(kernel_size, int) else [dilation * (x - 1) + 1 for x in kernel_size]
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    
    return tuple(padding) if isinstance(padding, list) else padding


class Conv(nn.Module):
    """标准卷积模块，Conv2d -> BatchNorm2d -> SiLU 激活"""

    default_act = nn.SiLU() # 默认激活函数 SiLU

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[Union[int, Tuple]] = None,
        groups: int = 1,
        dilation = 1,
        act = True
    ):
        """
        Args：
            in_channels（int）：输入通道数
            out_channels（int）：输出通道数
            kernel_size（int）：卷积核大小
            stride（int）：步长
            padding（int，Optional）：填充
            groups（int）：分组
            dilation（int）：卷积核膨胀【空洞】
            act（布尔值 | nn.Module）：激活函数
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 'same' if padding is None else padding, # padding = auto_padding(kernel_size=kernel_size, padding=padding, dilation=dilation),
            groups = groups,
            dilation = dilation,
            bias = False)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.act = nn.Identity()
        if act:
            self.act = act if isinstance(act, nn.Module) else self.default_act

    def forward(self, x) -> torch.Tensor:
        """激活"""
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x) -> torch.Tensor:
        """无激活"""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    标准 bottleneck 模块
    示意图：../assets/block_illustration/bottleneck.png
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernel_sizes: Tuple = (3, 3),
        expansion: float = 0.5
    ):
        """
        Args:
            in_channels（int）：输入通道数
            out_channels（int）：输出通道数
            stride（int）：步长
            groups（int）：分组
            kernel_sizes（Tuple）：两个卷积的卷积核大小
            expansion（float）：膨胀比例
        """
        super().__init__()
        _hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(
            in_channels = in_channels,
            out_channels = _hidden_channels,
            kernel_size = kernel_sizes[0],
            groups = groups)
        self.cv2 = Conv(
            in_channels = _hidden_channels,
            out_channels = out_channels,
            kernel_size = kernel_sizes[1],
            groups = groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    包含 2 个卷积层的 CSP Bottleneck 模块
    示意图：../assets/block_illustration/C2f.png
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
        groups: int = 1,
        shortcut: bool = False
    ):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 块的数量
            expansion (float): 膨胀比例
            groups (int): 分组
            shortcut (bool): 是否使用参差连接
        """
        super().__init__()
        self.hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels=in_channels, out_channels=2 * self.hidden_channels)
        self.cv2 = Conv(in_channels=(2 + num_blocks) * self.hidden_channels, out_channels=out_channels)
        self.module_list = nn.ModuleList(
            Bottleneck(
                in_channels = self.hidden_channels,
                out_channels = self.hidden_channels,
                shortcut = shortcut,
                groups = groups,
                kernel_sizes = ((3, 3), (3, 3)),
                expansion = 1.) for _ in range(num_blocks))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split(size=(self.hidden_channels, self.hidden_channels), dim=1))
        y.extend(m(y[-1]) for m in self.module_list)

        return self.cv2(torch.cat(y, 1))  


class C3(nn.Module):
    """
    包含 3 个卷积层的 CSP Bottleneck 模块
    示意图：../assets/block_illustration/C3.png
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int = 1,
            expansion: float = .5,
            groups: int = 1,
            shortcut: bool = False
        ):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 块的数量
            expansion (float): 膨胀比例
            groups (int): 分组
            shortcut (bool): 是否使用参差连接
        """
        super().__init__()
        _hidden_channels = int(out_channels * expansion)
        self.cv1 = Conv(in_channels=in_channels, out_channels=_hidden_channels)
        self.cv2 = Conv(in_channels=in_channels, out_channels=_hidden_channels)
        self.cv3 = Conv(in_channels=2 * _hidden_channels, out_channels=out_channels)
        self.module_list = nn.Sequential(
            *(Bottleneck(
               in_channels = _hidden_channels,
               out_channels = _hidden_channels,
               shortcut = shortcut,
               groups = groups,
               kernel_sizes = ((1, 1), (3, 3)),
               expansion = 1.) for _ in range(num_blocks)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.module_list(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """
    CSP 瓶颈模块，可定制内核大小
    示意图：../assets/block_illustration/C3k.png
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5,
        groups: int = 1,
        shortcut: bool = True,
        kernel_size: int = 3
    ):
        """
        Args:
            Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 块的数量
            expansion (float): 膨胀比例
            groups (int): 分组
            shortcut (bool): 是否使用参差连接
            kernel_size（int）：卷积核大小
        """
        super().__init__(in_channels, out_channels, num_blocks, expansion, groups, shortcut)
        _hidden_channels = int(out_channels * expansion)
        self.module_list = nn.Sequential(
            *(Bottleneck(
               in_channels = _hidden_channels,
               out_channels = _hidden_channels,
               shortcut = shortcut,
               groups = groups,
               kernel_sizes = (kernel_size, kernel_size),
               expansion = 1.) for _ in range(num_blocks)))


class C3k2(C2f):
    """
    C3k2 模块
    示意图：../assets/block_illustration/C3k2.png
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        c3k: bool = False,
        expansion: float = 0.5,
        groups: int = 1,
        shortcut: bool = True
    ):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 块的数量
            c3k (bool): 是否使用 C3k 块
            expansion (float): 膨胀比例
            groups (int): 分组
            shortcut (bool): 是否使用参差连接
        """
        super().__init__(in_channels, out_channels, num_blocks, expansion, groups, shortcut)
        if c3k:
            self.module_list = nn.ModuleList(
                C3k(
                    in_channels = self.hidden_channels,
                    out_channels = self.hidden_channels,
                    num_blocks = 2,
                    shortcut = shortcut,
                    groups = groups))
        else:
            self.module_list = nn.ModuleList(
                Bottleneck(
                    in_channels = self.hidden_channels,
                    out_channels = self.hidden_channels,
                    shortcut = shortcut,
                    groups = groups) for _ in range(num_blocks))


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1: int, c2: int, k: tuple[int, ...] = (5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            k (tuple): Kernel sizes for max pooling.
        """
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """
    空间金字塔池化-Fast
    示意图：../assets/block_illustration/SPPF.png
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        """等价于 SPP(k=(5, 9, 13))
        Args:
            in_channels (int): 输入维度.
            out_channels (int): 输出维度
            kernel_size (int): 核大小
        """
        super().__init__()
        _hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels=in_channels, out_channels=_hidden_channels)
        self.cv2 = Conv(in_channels=_hidden_channels * 4, out_channels=out_channels)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sequential pooling operations to input and return concatenated feature maps."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """
    自注意力模块
    示意图：../assets/block_illustration/Attention.png
    """
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        """
            dim (int): 特征维度
            num_heads (int): 注意力头
            attn_ratio (float): 关键维度的关注度比率
        """
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim 必须被 num_heads 整除"
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(in_channels=dim, out_channels=h, kernel_size=1, act=False)
        self.proj = Conv(in_channels=dim, out_channels=dim, kernel_size=1, act=False)
        self.pe = Conv(in_channels=dim, out_channels=dim, kernel_size=3, groups=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)

        return x


class PSABlock(nn.Module):
    """
    位置敏感注意力模块，封装了应用多头注意力和前馈神经网络层的功能
    示意图：../assets/block_illustration/PSABlock.png
    """
    def __init__(
        self,
        channels: int,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True
    ):
        """
        Args:
            channels（int）：输入和输出通道数
            attn_ratio（float）：维度的注意力比例
            num_heads（int）：注意力头的数量
            shortcut（bool）：是否使用参差连接
        """
        super().__init__()
        self.attn = Attention(channels, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            Conv(in_channels=channels, out_channels=channels * 2),
            Conv(in_channels=channels * 2, out_channels=channels, act=False))
        self.add = shortcut
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)

        return x


class C2PSA(nn.Module):
    """
    具有注意力机制的C2PSA模块，与 PSA 模块相同，但允许堆叠更多 PSABlock 模块
    示意图：../assets/block_illustration/C2PSA.png
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        expansion: float = 0.5
    ):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): 块的数量
            expansion (float): 膨胀比例
        """
        super().__init__()
        assert in_channels == out_channels
        self.hidden_channels = int(in_channels * expansion)
        self.cv1 = Conv(in_channels=in_channels, out_channels=2 * self.hidden_channels)
        self.cv2 = Conv(in_channels=2 * self.hidden_channels, out_channels=in_channels)
        self.module_list = nn.Sequential(
            *(PSABlock(
                self.hidden_channels,
                attn_ratio = 0.5,
                num_heads = self.hidden_channels // 64) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), dim=1))
        y[0] = self.module_list(y[0])

        return self.cv2(torch.cat(y, 1))
