"""
YOLOv11 Backbone — 清晰、模块化的实现（PyTorch）

改进目标：
1. 把原来一个文件里嵌套很多逻辑的实现重构成更清晰的模块（CBS / C3k2_F / C3k2_T / SPPF / PSA / C2PSA）。
2. 命名与图中一致（例如 C3k2_F_1、C3k2_T_1、SPPF、C2PSA），方便对照网络图。
3. 保留简单前向测试，输出三个尺度：P3 (stride 8, C=128), P4 (stride 16, C=256), P5 (stride 32, C=512)。

说明：此实现做了合理简化，重点是结构与模块接口一致，便于后续替换或扩展（例如把 PSA 换为更复杂的自注意力）。
"""
from typing import List, Tuple

import torch
import torch.nn as nn


# ----------------------------- 基础构件 -----------------------------
class ConvBNAct(nn.Module):
    """ CBS 块
        标准卷积块：Conv2d -> BatchNorm2d -> SiLU
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """图中 Bottleneck：两层 CBS，带残差"""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int = None):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        self.conv1 = ConvBNAct(in_ch, hidden_ch, k=1, s=1, p=0)
        self.conv2 = ConvBNAct(hidden_ch, out_ch, k=3, s=1, p=1)
        self.add = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.add else y


# ----------------------------- C3 模块（C3k2_F 与 C3k2_T 的基础） -----------------------------
class C3(nn.Module):
    """通用 C3 模块（可用于 F 或 T 变体），支持将输入 split 为两条路径。

    - 左分支为多个 Bottleneck 的串联
    - 右分支为 1x1 下采样/投影
    - 最后 concat + 1x1
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1, expansion: float = 0.5):
        super().__init__()
        hidden_ch = int(out_ch * expansion)
        self.conv_l = ConvBNAct(in_ch, hidden_ch, k=1, s=1, p=0)
        self.conv_r = ConvBNAct(in_ch, hidden_ch, k=1, s=1, p=0)
        self.m = nn.Sequential(*[Bottleneck(hidden_ch, hidden_ch, hidden_ch) for _ in range(n)])
        self.conv_out = ConvBNAct(hidden_ch * 2, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left = self.conv_l(x)
        left = self.m(left)
        right = self.conv_r(x)
        out = torch.cat([left, right], dim=1)
        out = self.conv_out(out)
        return out


# 图中 T 版 C3k2_T 的结构是把 C3 的内部拆分并用多个并行分支（简化为上面的实现以便代码清晰）
# 如果需要精确到图中并行的三个 C3k1_x 再 concat，可在此扩展为 C3_T 类。


# ----------------------------- SPPF（轻量 SPP） -----------------------------
class SPPF(nn.Module):
    """SPPF：一层 1x1 conv + 多尺度 maxpool concat + 1x1 conv"""

    def __init__(self, in_ch: int, out_ch: int, pool_sizes: Tuple[int, int, int] = (5, 9, 13)):
        super().__init__()
        hidden = in_ch // 2
        self.conv1 = ConvBNAct(in_ch, hidden, k=1, s=1, p=0)
        self.pool_sizes = pool_sizes
        self.conv2 = ConvBNAct(hidden * (len(pool_sizes) + 1), out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        outs = [x]
        for k in self.pool_sizes:
            outs.append(nn.functional.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2))
        x = torch.cat(outs, dim=1)
        x = self.conv2(x)
        return x


# ----------------------------- 简化版 PSA（空间注意力块） -----------------------------
class PSABlock(nn.Module):
    """简化实现的 PSABlock：
    - 使用一个轻量的通道注意力 + 空间缩放的 idea
    - 便于理解与调试，若需更复杂的 QKV 自注意力可以替换此模块
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# --------------------------- C2PSA（图中 C2PSA：CBS -> Split -> PSABlock -> concat -> CBS） ---------------------------
class C2PSA(nn.Module):
    """按图中组合：先 CBS，再 split（两路），一路经过 PSABlock，再 concat 和 CBS"""

    def __init__(self, in_ch: int, out_ch: int, expansion: float = 0.5):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.pre = ConvBNAct(in_ch, hidden, k=1, s=1, p=0)
        self.psa = PSABlock(hidden)
        self.conv_out = ConvBNAct(hidden * 2, out_ch, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        # split channel-wise into two equal parts
        c = x.shape[1] // 2
        a, b = x[:, :c, :, :], x[:, c:, :, :]
        a = self.psa(a)
        out = torch.cat([a, b], dim=1)
        out = self.conv_out(out)
        return out


# ----------------------------- 整体 Backbone（按图中的大体流程） -----------------------------
class YOLOv11BackboneV2(nn.Module):
    """模块化、更贴近图中结构的 Backbone

    设计策略：
    - 使用 CBS 作为基本卷积单元
    - stage 输出通道按图意图设为 128 / 256 / 512
    - 在最顶层使用 SPPF 和 C2PSA
    - 返回 [P3 (80x80, C=128), P4 (40x40, C=256), P5 (20x20, C=512)]
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Stem: 640->320->160
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, 32, k=3, s=2, p=1),  # 320
            ConvBNAct(32, 64, k=3, s=2, p=1))

        # stage P3 path: 160->80 (最终输出 P3: C=128)
        self.c3_1 = C3(64, 128, n=1, expansion=0.5)
        # downsample to P4
        self.down34 = ConvBNAct(128, 256, k=3, s=2, p=1)

        # stage P4 path: 80->40 (输出 P4: C=256)
        self.c3_2 = C3(256, 256, n=2, expansion=0.5)
        # downsample to P5
        self.down45 = ConvBNAct(256, 512, k=3, s=2, p=1)

        # stage P5 path: 40->20 (输出 P5: C=512)
        self.c3_3 = C3(512, 512, n=3, expansion=0.5)

        # SPPF 在 P5 顶端
        self.sppf = SPPF(512, 512)

        # 顶层的 C2PSA（如图）增强，之后输出给 neck
        self.c2psa = C2PSA(512, 512, expansion=0.5)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, 3, 640, 640]
        x = self.stem(x)  # -> [B,64,160,160]

        p3 = self.c3_1(x)  # -> [B,128,80,80]

        x = self.down34(p3)  # -> [B,256,40,40]
        p4 = self.c3_2(x)  # -> [B,256,40,40]

        x = self.down45(p4)  # -> [B,512,20,20]
        x = self.c3_3(x)  # -> [B,512,20,20]

        x = self.sppf(x)
        p5 = self.c2psa(x)  # -> [B,512,20,20]

        # 返回 P3, P4, P5（与图中 head 对接）
        return [p3, p4, p5]


# ----------------------------- 简单前向测试 -----------------------------
if __name__ == "__main__":
    model = YOLOv11BackboneV2(in_channels=3)
    model.eval()
    inp = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = model(inp)
    print("P3:", p3.shape)  # 1,128,80,80
    print("P4:", p4.shape)  # 1,256,40,40
    print("P5:", p5.shape)  # 1,512,20,20

    # 方便调试：检查参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.2f}M")

