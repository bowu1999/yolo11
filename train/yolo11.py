from typing import List, Tuple, Optional, Any, Dict, Dict, Sequence, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torch.distributed as dist

import gc
import os
import timm
import argparse
from PIL import Image

from mmengine.model import BaseModel
from mmengine.registry import DATASETS, MODELS, METRICS, FUNCTIONS
from mmengine.evaluator import BaseMetric
from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.dist import init_dist

from haua.datasets import COCODetectionDataset, coco_collate, get_train_transforms
from haua.models import Yolo11_train
from haua.losses import YOLOv10Loss


class YOLOCOCO(COCODetectionDataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        return_masks: bool = False
    ):
        super().__init__(
            root = root,
            ann_file = ann_file,
            transforms = get_train_transforms(640, auto=False),
            return_masks = return_masks)

class TrainYOLO11(BaseModel):
    """MMEngine 封装的多任务模型"""
    def __init__(
        self,
        backbone_config: dict,
        loss_config: dict,
    ):
        super().__init__()
        self.backbone = Yolo11_train(**backbone_config)
        self.loss_module = YOLOv10Loss(**loss_config)

    def forward(self, inputs, data_samples=None, mode: str = "tensor"): # type: ignore
        """
        Args:
            batch_inputs: 输入张量 (B, ...)
            data_samples: list[dict]，每个dict包含标签
            mode: "tensor" | "loss" | "predict"
        """
        preds = self.backbone(inputs)
        if mode == "tensor":
            return preds
        elif mode == "loss":
            assert data_samples is not None, "训练时必须提供 data_samples"
            return self.loss(preds, data_samples) # type: ignore
        # elif mode == "predict":
        #     return self.predict(preds)
        else:
            raise ValueError(f"Invalid mode {mode}")

    def loss(self,
        outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        data_samples: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        loss = self.loss_module(outputs, data_samples)

        return loss

    # def predict(self, outputs):
    #     B = outputs['gender'].shape[0]
    #     results = []
    #     for i in range(B):
    #         results.append(self.output_parsing(outputs['gender'][i], outputs['age'][i]))

    #     return results


def cleanup():
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()

def main():
    try:
        parser = argparse.ArgumentParser(description='Train/Test script for HWP model')
        parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
        args = parser.parse_args()
        init_dist('pytorch')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        config_file_path = args.config
        print(f"Using config file: {config_file_path}")
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        config = Config.fromfile(config_file_path)
        if not hasattr(config, 'model_wrapper_cfg'):
            config.model_wrapper_cfg = dict(
                type='MMDistributedDataParallel',
                find_unused_parameters=False,
                broadcast_buffers=False)
        config.launcher = 'pytorch'
        runner = Runner.from_cfg(config)
        runner.train()
    finally:
        cleanup()


if __name__ == '__main__':
    DATASETS.register_module(module=YOLOCOCO)
    MODELS.register_module(module=TrainYOLO11)
    FUNCTIONS.register_module(module=coco_collate)
    main()