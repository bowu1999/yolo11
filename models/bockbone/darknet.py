import torch
import torch.nn as nn

from blocks import Conv, C3k2, SPPF, C2PSA


_MODEL_CONFIGS = {
    "n": {"depth_mult": 0.33, "width_mult": 0.25, "stem_channels": 16, "num_blocks": [1, 2, 2, 1]},
    "s": {"depth_mult": 0.33, "width_mult": 0.50, "stem_channels": 32, "num_blocks": [1, 2, 2, 1]},
    "m": {"depth_mult": 0.67, "width_mult": 0.75, "stem_channels": 48, "num_blocks": [2, 4, 4, 2]},
    "l": {"depth_mult": 1.00, "width_mult": 1.00, "stem_channels": 64, "num_blocks": [3, 6, 6, 3]},
    "x": {"depth_mult": 1.33, "width_mult": 1.25, "stem_channels": 80, "num_blocks": [4, 8, 8, 4]}}


def make_divisible(x, divisor=8):
    """将通道数调整为8的倍数"""
    return int((x + divisor / 2) // divisor * divisor)


def build_layer(in_channels, out_channels, num_blocks, c3k=True, downsample=True):
    """构建 Darknet 的层，包含下采样 Conv + C3k2 块堆叠"""
    layers = []
    if downsample:
        layers.append(
            Conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2))
    layers.append(
        C3k2(
            in_channels = in_channels if not downsample else out_channels,
            out_channels = out_channels,
            num_blocks = num_blocks,
            c3k = c3k))
    return nn.Sequential(*layers)


class Darknet(nn.Module):
    base_channels = [64, 128, 256, 512] # yolo11 L 表示标准
    def __init__(self, model_type="s", in_channels=3):
        super().__init__()
        assert model_type in _MODEL_CONFIGS, f"Unsupported type: {model_type}"
        cfg = _MODEL_CONFIGS[model_type]
        width_mult, stem_channels = cfg["width_mult"], cfg["stem_channels"]
        self.model_channels = [make_divisible(c * width_mult) for c in self.base_channels]
        self.stem = nn.Sequential(
            Conv(in_channels=in_channels, out_channels=stem_channels, kernel_size=3, stride=1),
            Conv(
                in_channels = stem_channels,
                out_channels = self.model_channels[0],
                kernel_size = 3,
                stride = 2))
        self.layer1 = build_layer(
            in_channels = self.model_channels[0],
            out_channels = self.model_channels[0],
            num_blocks = cfg["num_blocks"][0],
            c3k = False,
            downsample = False)
        self.layer2 = build_layer(
            in_channels = self.model_channels[0],
            out_channels = self.model_channels[1],
            num_blocks = cfg["num_blocks"][1],
            c3k = False)
        self.layer3 = build_layer(
            in_channels = self.model_channels[1],
            out_channels = self.model_channels[2],
            num_blocks = cfg["num_blocks"][2])
        self.layer4 = build_layer(
            in_channels = self.model_channels[2],
            out_channels = self.model_channels[3],
            num_blocks = cfg["num_blocks"][3])
        self.sppf = SPPF(in_channels=self.model_channels[3], out_channels=self.model_channels[3])
        self.c2psa = C2PSA(in_channels=self.model_channels[3], out_channels=self.model_channels[3])

        self.out_channels = self.model_channels[-3:]

    def forward(self, x):
        stem_out = self.stem(x)
        layer1_out = self.layer1(stem_out)
        layer2_out = self.layer2(layer1_out)
        layer3_out = self.layer3(layer2_out)
        layer4_out = self.layer4(layer3_out)
        sppf_out = self.sppf(layer4_out)
        final_out = self.c2psa(sppf_out)

        return layer2_out, layer3_out, final_out