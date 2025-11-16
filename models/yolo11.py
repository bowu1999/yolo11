from typing import List, Tuple, Any, Dict, Union, Optional, Callable

from ._basemodel import BaseModel
from .backbone import Darknet
from .neck import C3k2PAN
from .head import YOLODetector

from ._utils import make_divisible


_MODEL_CONFIGS = {
    "n": {
        "depth_mult": .5, "width_mult": .25, "max_channels": 1024,
        "backbone": {"block_args": [(False, .25), (False, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 80}},
    "s": {
        "depth_mult": .5, "width_mult": .5, "max_channels": 1024,
        "backbone": {"block_args": [(False, .25), (False, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 128}},
    "m": {
        "depth_mult": .5, "width_mult": 1., "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 256}},
    "l": {
        "depth_mult": 1., "width_mult": 1., "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 64, "classify_hidden_channels": 256}},
    "x": {
        "depth_mult": 1., "width_mult": 1.50, "max_channels": 512,
        "backbone": {"block_args": [(True, .25), (True, .25), (True,), (True,)]},
        "neck":{},
        "head": {"locate_hidden_channels": 96, "classify_hidden_channels": 384}}}


class Yolo11(BaseModel):
    
    base_backbone_layer_channels = [128, 256, 512, 512, 1024]
    base_backbone_num_blocks = [2, 2, 2, 2, 2]
    base_neck_out_channels = [256, 512, 1024]

    def __init__(
        self,
        model_type: str = "l",
        num_classes: int = 80,
        custom_postprocess: Optional[Callable[[Any], Any]] = None
    ):
        assert model_type in _MODEL_CONFIGS, f"Unsupported type: {model_type}"
        cfg = _MODEL_CONFIGS[model_type]
        self.width_mult = cfg["width_mult"]
        self.depth_mult = cfg["depth_mult"]
        self.max_channels = cfg["max_channels"]
        backbone_block_args = cfg["backbone"]["block_args"]
        head_locate_hidden_channels = cfg["head"]["locate_hidden_channels"]
        head_classify_hidden_channels = cfg["head"]["classify_hidden_channels"]
        backbone_layer_channels = [
            make_divisible(c * self.width_mult) for c in self.base_backbone_layer_channels]
        backbone_layer_channels = self._channel_trimming(backbone_layer_channels)
        backbone_num_blocks = [int(n * self.depth_mult) for n in self.base_backbone_num_blocks]
        neck_out_channels = [
            make_divisible(c * self.width_mult) for c in self.base_neck_out_channels]
        neck_out_channels = self._channel_trimming(neck_out_channels)
        module_configs = {
            "backbone": {
                "layer_channels": backbone_layer_channels,
                "num_blocks": backbone_num_blocks,
                "block_args": backbone_block_args},
            "neck": {
                "in_channels": backbone_layer_channels[-3:],
                "out_channels": neck_out_channels},
            "head": {
                "in_channels_list": neck_out_channels,
                "num_classes": num_classes,
                "locate_hidden_channels": head_locate_hidden_channels,
                "classify_hidden_channels": head_classify_hidden_channels}}
        super().__init__(Darknet, C3k2PAN, YOLODetector, custom_postprocess, module_configs)
    
    def _channel_trimming(self, channels: List[int]) -> List[int]:
        if channels[-1] > self.max_channels:
            channels[-1] = int(channels[-1] // 2)
        
        return channels
