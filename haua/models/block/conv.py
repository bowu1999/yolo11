from typing import List, Optional, Tuple, Union, Optional, Callable

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_padding(
    kernel_size: Union[int, List[int], Tuple[int, ...]],
    padding: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    dilation: int = 1,
) -> Union[int, Tuple[int, ...]]:
    if padding is None:
        if isinstance(kernel_size, int):
            kernel_sizes: Tuple[int, ...] = (kernel_size, kernel_size)
        else:
            kernel_sizes = tuple(kernel_size)
        computed = tuple((((k - 1) * dilation + 1) // 2) for k in kernel_sizes)
        return computed[0] if len(computed) == 1 else computed

    return padding


def _get_act(act: Optional[Union[str, bool, nn.Module]] = "silu") -> Callable[[], nn.Module]:
    if act is None or act is False:
        return lambda: nn.Identity()
    if isinstance(act, nn.Module):
        return lambda: act
    name = str(act).lower()
    if name in ("relu", "relu6"):
        return lambda: nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return lambda: nn.SiLU(inplace=True)
    if name == "gelu":
        return lambda: nn.GELU()
    if name == "leakyrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
    raise ValueError(f"Unsupported act: {act}")


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    将 Conv2d + BatchNorm2d 融合成一个带 bias 的 Conv2d（in-place style 返回新 conv）。
    保证所有参数使用命名参数构造。
    """
    assert isinstance(conv, nn.Conv2d), "conv 必须为 nn.Conv2d"
    assert isinstance(bn, nn.BatchNorm2d), "bn 必须为 nn.BatchNorm2d"

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        device=conv.weight.device,
        dtype=conv.weight.dtype)
    bn_weight = bn.weight
    bn_bias = bn.bias
    bn_running_mean = bn.running_mean
    bn_running_var = bn.running_var
    bn_eps = bn.eps
    scale = bn_weight / torch.sqrt(bn_running_var + bn_eps)
    fused_conv.weight.data = conv.weight.data * scale.reshape((-1, 1, 1, 1))
    fused_conv.bias.data = bn_bias - bn_running_mean * scale

    return fused_conv


class ConvBNAct(nn.Module):
    """
    Conv -> BN -> Act 模块，提供 fuse() 将 BN 融合进 Conv。
    所有构造参数在 nn 模块创建时使用命名参数。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        groups: int = 1,
        dilation: int = 1,
        act: Union[bool, str, nn.Module] = "silu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=auto_padding(kernel_size, padding, dilation),
            dilation=dilation,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-3, momentum=0.03)
        act_ctor = _get_act(act)
        self.act = act_ctor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out

    def forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.act(out)

        return out

    def fuse(self) -> "ConvBNAct":
        """
        将 bn 融合进 conv，并把 forward 替换为 fused 版本（原地修改）。
        返回 self，以便链式调用。
        """
        if hasattr(self, "bn") and isinstance(self.bn, nn.BatchNorm2d):
            self.conv = fuse_conv_bn(conv=self.conv, bn=self.bn)
            delattr(self, "bn")
            self.forward = self.forward_fused

        return self


class ChannelLayerNorm(nn.Module):
    """对 (B, C, H, W) 的特征在 channels 维度上做 LayerNorm（ConvNeXt 风格）"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 3, 1) # x: (B, C, H, W) -> (B, H, W, C)
        x_norm = self.norm(x_perm)

        return x_norm.permute(0, 3, 1, 2)


class UniversalStem(nn.Module):
    """
    通用 Stem 实现，支持多种风格：
    stem_type in {
        "standard","deep","convnext","efficient","hybrid","inception","patch","yolo"}
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        stem_type: str = "convnext",
        act: Union[bool, str, nn.Module] = "silu",
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        deep_layers: int = 3,
        hybrid_large_kernels: Optional[List[int]] = None,
        inception_branches: int = 3,
        last_pool: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem_type = stem_type.lower()
        self.act = act
        self.norm_layer = norm_layer or \
            (lambda num_features: nn.BatchNorm2d(num_features=num_features))
        self._construct_stem(
            deep_layers=deep_layers,
            hybrid_large_kernels=hybrid_large_kernels,
            inception_branches=inception_branches,
            last_pool=last_pool)

    def _construct_stem(
        self,
        deep_layers: int,
        hybrid_large_kernels: Optional[List[int]],
        inception_branches: int,
        last_pool: bool,
    ) -> None:
        if self.stem_type == "standard":
            # ResNet 风格：7x7 stride2 -> BN -> Act -> optional MaxPool
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    kernel_size = 7,
                    stride = 2,
                    padding = 3,
                    bias = False),
                self.norm_layer(self.out_channels),
                _get_act(self.act)(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if last_pool else nn.Identity())
        elif self.stem_type == "deep":
            # 多个 3x3 conv，首层 stride=2
            layers: List[nn.Module] = []
            cur_channels = self.in_channels
            for i in range(int(deep_layers)):
                stride_val = 2 if i == 0 else 1
                next_channels = self.out_channels \
                    if i == (deep_layers - 1) else max(cur_channels, self.out_channels // 2)
                layers.append(
                    ConvBNAct(
                        in_channels=cur_channels,
                        out_channels=next_channels,
                        kernel_size=3,
                        stride=stride_val,
                        padding=None,
                        groups=1,
                        dilation=1,
                        act=self.act))
                cur_channels = next_channels
            if last_pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            self.block = nn.Sequential(*layers)
        elif self.stem_type == "convnext":
            # ConvNeXt 风格：一次大降采样（4x4, stride=4） + ChannelLayerNorm + Act
            self.block = nn.Sequential(
                nn.Conv2d(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    kernel_size = 4,
                    stride = 4,
                    padding = 0,
                    bias = False),
                ChannelLayerNorm(num_channels=self.out_channels),
                _get_act(self.act)())
        elif self.stem_type == "efficient":
            # EfficientNet 风格：3x3 stride2 -> BN -> Act（主干使用 MBConv/Fused）
            self.block = nn.Sequential(
                ConvBNAct(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels,
                    kernel_size = 3,
                    stride = 2,
                    padding = None,
                    groups = 1,
                    dilation =1,
                    act = self.act))
        elif self.stem_type == "hybrid":
            # Hybrid: small path + multiple large-kernel paths (提前构建好所有分支)
            if hybrid_large_kernels is None:
                hybrid_large_kernels = [31]
            small_path = nn.Sequential(
                ConvBNAct(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels // 2,
                    kernel_size = 3,
                    stride = 2,
                    padding = None,
                    groups = 1,
                    dilation = 1,
                    act = self.act),
                ConvBNAct(
                    in_channels = self.out_channels // 2,
                    out_channels = self.out_channels // 2,
                    kernel_size = 3,
                    stride = 1,
                    padding = None,
                    groups = 1,
                    dilation = 1,
                    act = self.act))
            large_paths: nn.ModuleDict = nn.ModuleDict()
            for idx, k in enumerate(hybrid_large_kernels):
                pad_val = (k - 1) // 2
                # 为了效率采用 depthwise large kernel + pointwise proj
                large_paths[f"large_{idx}"] = nn.Sequential(
                    nn.Conv2d(
                        in_channels = self.in_channels,
                        out_channels = self.out_channels // 2,
                        kernel_size = k,
                        stride = 4,
                        padding = pad_val,
                        groups = 1,
                        bias = False),
                    self.norm_layer(self.out_channels // 2),
                    _get_act(self.act)())
            # 统一 fuse -> reduce to out_channels
            fuse_reducer = ConvBNAct(
                in_channels = (self.out_channels // 2) * (1 + len(large_paths)),
                out_channels = self.out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                groups = 1,
                dilation = 1,
                act = self.act)
            self.hybrid_small = small_path
            self.hybrid_large = large_paths
            self.hybrid_fuse = fuse_reducer
        elif self.stem_type == "inception":
            # Inception-like: 并行分支 -> concat -> 1x1 fuse
            branches: List[nn.Module] = []
            # branch A: 3x3 stride2
            branches.append(
                ConvBNAct(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels // 2,
                    kernel_size = 3,
                    stride = 2,
                    padding = None,
                    act = self.act))
            # branch B: 1x1 -> 3x3 stride2
            branches.append(nn.Sequential(
                ConvBNAct(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels // 4,
                    kernel_size = 1,
                    stride = 1,
                    padding = 0,
                    act = self.act),
                ConvBNAct(
                    in_channels = self.out_channels // 4,
                    out_channels = self.out_channels // 4,
                    kernel_size = 3,
                    stride = 2,
                    padding = None,
                    act = self.act)))
            branches.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            for _ in range(max(0, int(inception_branches) - 3)):
                branches.append(
                    ConvBNAct(
                        in_channels = self.in_channels,
                        out_channels = self.out_channels // inception_branches,
                        kernel_size = 3,
                        stride = 2,
                        padding = None,
                        act=self.act))
            self.inception_branches = nn.ModuleList(branches)
            self.inception_fuse = ConvBNAct(
                in_channels = sum([self._branch_out_channels(b) for b in self.inception_branches]),
                out_channels = self.out_channels,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                act=self.act)
        elif self.stem_type == "patch":
            # Patch embedding 风格：4x4 stride4 -> LayerNorm(Channels) -> Act
            self.patch_conv = nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=4, stride=4, padding=0, bias=False)
            self.patch_norm = ChannelLayerNorm(num_channels=self.out_channels)
            self.patch_act = _get_act(self.act)()
        elif self.stem_type == "yolo":
            # YOLO 风格轻量 stem：两个 3x3 conv，首层 stride=2
            self.block = nn.Sequential(
                ConvBNAct(
                    in_channels = self.in_channels,
                    out_channels = self.out_channels // 2,
                    kernel_size = 3,
                    stride = 2,
                    padding = None,
                    act = self.act),
                ConvBNAct(
                    in_channels = self.out_channels // 2,
                    out_channels = self.out_channels,
                    kernel_size = 3,
                    stride = 1,
                    padding = None,
                    act = self.act))
        else:
            raise ValueError(f"Unsupported stem_type: {self.stem_type}")

    def _branch_out_channels(self, branch: nn.Module) -> int:
        """辅助：估计分支输出通道数（用于 inception fuse 构造），仅对常见分支结构可靠。"""
        # 常见情况下 branch 最后有 ConvBNAct，可以读取其 conv.out_channels
        if isinstance(branch, ConvBNAct):
            return int(branch.conv.out_channels)
        if isinstance(branch, nn.Sequential):
            for m in reversed(branch):
                if isinstance(m, ConvBNAct):
                    return int(m.conv.out_channels)
            # fallback：尝试第一个 Conv2d
            for m in branch:
                if isinstance(m, nn.Conv2d):
                    return int(m.out_channels)
        if isinstance(branch, nn.MaxPool2d):
            # MaxPool channels == input channels (推断为 in_channels)
            return int(self.in_channels)

        # 保守返回 out_channels // n（caller 根据实际情况调整）
        return int(self.out_channels // max(1, len(list(self.inception_branches) \
            if hasattr(self, "inception_branches") else [1])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stem_type in ("standard", "deep", "convnext", "efficient", "yolo"):
            return self.block(x)
        if self.stem_type == "patch":
            x = self.patch_conv(x)
            x = self.patch_norm(x)
            x = self.patch_act(x)
            return x
        if self.stem_type == "hybrid":
            outs: List[torch.Tensor] = []
            outs.append(self.hybrid_small(x))
            for key in self.hybrid_large:
                outs.append(self.hybrid_large[key](x))
            # 统一上采样到最大空间尺寸以便 concat（通常 large path stride 为 4，small path 2，需对齐）
            sizes = [o.shape[2:] for o in outs]
            max_h = max([s[0] for s in sizes])
            max_w = max([s[1] for s in sizes])
            aligned: List[torch.Tensor] = []
            for o in outs:
                if (o.shape[2], o.shape[3]) != (max_h, max_w):
                    o = F.interpolate(
                        input=o, size=(max_h, max_w), mode="bilinear", align_corners=False)
                aligned.append(o)
            fused = torch.cat(aligned, dim=1)
            out = self.hybrid_fuse(fused)
            return out
        if self.stem_type == "inception":
            outs: List[torch.Tensor] = []
            for branch in self.inception_branches:
                if isinstance(branch, nn.MaxPool2d):
                    outs.append(branch(x))
                else:
                    outs.append(branch(x))
            concat = torch.cat(outs, dim=1)
            out = self.inception_fuse(concat)
            return out

        raise RuntimeError("Unhandled stem_type in forward")

    def fuse(self) -> "UniversalStem":
        """
        递归融合 internal ConvBNAct 中的 BN（训练结束后用于加速推理）。
        返回 self 以便链式使用。
        """
        def _fuse_module(m: nn.Module) -> None:
            for name, child in list(m.named_children()):
                if isinstance(child, ConvBNAct):
                    child.fuse()
                else:
                    _fuse_module(child)
        _fuse_module(self)

        return self

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
            in_channels (int):输入通道数
            out_channels (int):输出通道数
            stride (int):步长
            groups (int):分组
            kernel_sizes（Tuple）：两个卷积的卷积核大小
            expansion（float）：膨胀比例
        """
        super().__init__()
        _hidden_channels = int(out_channels * expansion)
        self.cba1 = ConvBNAct(
            in_channels = in_channels,
            out_channels = _hidden_channels,
            kernel_size = kernel_sizes[0],
            groups = groups)
        self.cba2 = ConvBNAct(
            in_channels = _hidden_channels,
            out_channels = out_channels,
            kernel_size = kernel_sizes[1],
            groups = groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cba2(self.cba1(x)) if self.add else self.cba2(self.cba1(x))


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
        self.front_cba = ConvBNAct(in_channels=in_channels, out_channels=2 * self.hidden_channels)
        self.back_cba = ConvBNAct(
            in_channels = (num_blocks + 2) * self.hidden_channels,
            out_channels = out_channels)
        self.module_list = nn.ModuleList(
            Bottleneck(
                in_channels = self.hidden_channels,
                out_channels = self.hidden_channels,
                shortcut = shortcut,
                groups = groups,
                kernel_sizes = ((3, 3), (3, 3)),
                expansion = 1.) for _ in range(num_blocks))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.front_cba(x)
        x1, x2 = x.split((self.hidden_channels, self.hidden_channels), dim=1)
        xs = [x1, x2]
        for m in self.module_list:
            x2 = m(x2)
            xs.append(x2)

        return self.back_cba(torch.cat(xs, 1))


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
        self.left_cba = ConvBNAct(in_channels=in_channels, out_channels=_hidden_channels)
        self.right_cba = ConvBNAct(in_channels=in_channels, out_channels=_hidden_channels)
        self.back_cba = ConvBNAct(in_channels=2 * _hidden_channels, out_channels=out_channels)
        self.module_list = nn.ModuleList(
            Bottleneck(
               in_channels = _hidden_channels,
               out_channels = _hidden_channels,
               shortcut = shortcut,
               groups = groups,
               kernel_sizes = ((1, 1), (3, 3)),
               expansion = 1.) for _ in range(num_blocks))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.left_cba(x)
        for m in self.module_list:
            x1 = m(x1)
        x2 = self.right_cba(x)

        return self.back_cba(torch.cat((x1, x2), 1))


def convert_kernel_size(a):
    """转换卷积核参数格式"""
    return ((a, a), (a, a)) if isinstance(a, int) else \
           a if isinstance(a[0], tuple) else \
           tuple((x, x) for x in a)


class C3k(C3):
    """
    CSP Bottleneck 模块，可定制内核大小
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
        kernel_size: Union[int, Tuple[int, int]] = (3, 3)
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
            kernel_size (int):卷积核大小
        """
        super().__init__(in_channels, out_channels, num_blocks, expansion, groups, shortcut)
        _hidden_channels = int(out_channels * expansion)
        self.module_list = nn.ModuleList(
            Bottleneck(
               in_channels = _hidden_channels,
               out_channels = _hidden_channels,
               shortcut = shortcut,
               groups = groups,
               kernel_sizes = convert_kernel_size(kernel_size),
               expansion = 1.) for _ in range(num_blocks))


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
                    groups = groups) for _ in range(num_blocks))
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
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c_ * (len(k) + 1), c2, 1, 1)
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
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=_hidden_channels)
        self.cv2 = ConvBNAct(in_channels=_hidden_channels * 4, out_channels=out_channels)
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
        self.qkv = ConvBNAct(in_channels=dim, out_channels=h, kernel_size=1, act=False)
        self.proj = ConvBNAct(in_channels=dim, out_channels=dim, kernel_size=1, act=False)
        self.pe = ConvBNAct(in_channels=dim, out_channels=dim, kernel_size=3, groups=dim, act=False)

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
            channels (int):输入和输出通道数
            attn_ratio（float）：维度的注意力比例
            num_heads (int):注意力头的数量
            shortcut（bool）：是否使用参差连接
        """
        super().__init__()
        self.attn = Attention(channels, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            ConvBNAct(in_channels=channels, out_channels=channels * 2),
            ConvBNAct(in_channels=channels * 2, out_channels=channels, act=False))
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
        self.cv1 = ConvBNAct(in_channels=in_channels, out_channels=2 * self.hidden_channels)
        self.cv2 = ConvBNAct(in_channels=2 * self.hidden_channels, out_channels=in_channels)
        self.module_list = nn.Sequential(
            *(PSABlock(
                self.hidden_channels,
                attn_ratio = 0.5,
                num_heads = self.hidden_channels // 64) for _ in range(num_blocks)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.hidden_channels, self.hidden_channels), dim=1))
        y[0] = self.module_list(y[0])

        return self.cv2(torch.cat(y, 1))
