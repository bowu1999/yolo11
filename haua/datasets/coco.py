from pathlib import Path
from typing import Callable, List, Tuple, Optional, Dict, Any

import json
import math
import random

from PIL import Image, ImageOps
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

try:
    from pycocotools.coco import COCO
except Exception as e:
    COCO = None
    # 如果没有 pycocotools，Dataset 的初始化会报错，提示安装 pycocotools


# 在文件顶部（imports 之后）添加 COCO id -> 0..79 的映射
# 原始 COCO 的 80 类在 annotation 中对应的 category_id（不连续）
COCO80_ORIG_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90]
# map original id -> contiguous 0..79
COCO_ID_TO_80 = {orig_id: i for i, orig_id in enumerate(COCO80_ORIG_IDS)}
# optionally inverse mapping (0..79 -> orig id)
COCO_80_TO_ORIG = {i: orig for i, orig in enumerate(COCO80_ORIG_IDS)}

# 标准 COCO 80 类名，索引 0..79 对应上面的 COCO80_ORIG_IDS 映射顺序
coco80_names = [
 "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
 "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
 "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
 "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
 "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
 "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
 "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
 "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """COCO 的 bbox 是 [x, y, w, h] -> 转为 [x1, y1, x2, y2]"""
    boxes = boxes.copy().astype(np.float32)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    return boxes


def clip_boxes(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


class ToTensor:
    def __call__(self, image, target):
        # image: PIL -> Tensor [C,H,W] 0-1
        image = F.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            res = t(image, target)
            if isinstance(res, (tuple, list)):
                if len(res) == 2:
                    image, target = res
                elif len(res) == 1:
                    image = res[0]
                else:
                    raise RuntimeError(f"Transform {t} returned unexpected tuple length {len(res)}")
            else:
                image = res
        return image, target


class InferCompose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, image, target=None):
        was_none = target is None
        if was_none:
            target = {}
        for t in self.transforms:
            res = t(image, target)
            if isinstance(res, (tuple, list)):
                if len(res) == 2:
                    image, target = res
                elif len(res) == 1:
                    image = res[0]
                else:
                    raise RuntimeError(f"Transform {t} returned unexpected tuple length {len(res)}")
            else:
                image = res
        if was_none:
            return image
        return image, target


class ImageOnlyWrapper:
    def __init__(self, fn: Callable):
        self.fn = fn

    def __call__(self, image, target=None):
        out = self.fn(image)
        if isinstance(out, (tuple, list)):
            image = out[0]
        else:
            image = out
        return image, target
    

class LetterBox:
    """
    调整图片大小并填充至目标图片尺寸，同时保持宽高比

    Args:
        img_size: int or (h, w). if int -> square (img_size, img_size)
        stride: int padding divisible by stride (通常为模型最大 stride，如 32)
        auto: 如果为真，则使填充形状可被步长整除（最小填充）
        scaleup: 允许放大小图像
        color: pad color (0-255) or tuple (R,G,B)

    Returns PIL.Image 和 更新目标（调整目标框）
    """
    def __init__(self, img_size=640, stride=32, auto=True, scaleup=True, color=114):
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.scaleup = scaleup
        if isinstance(color, int):
            self.color = (color, color, color)
        else:
            self.color = color

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None):
        orig_w, orig_h = image.size  # PIL: (w, h)
        target_h, target_w = self.img_size[0], self.img_size[1]
        # compute scale to fit in target while preserving aspect ratio
        r = min(target_w / orig_w, target_h / orig_h)
        if not self.scaleup:
            r = min(r, 1.0)
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        # resize
        if (orig_w, orig_h) != (new_w, new_h):
            image = image.resize((new_w, new_h), resample=Image.BILINEAR) # type: ignore
        # compute padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        # if auto: make padding divisible by stride
        if self.auto and self.stride:
            pad_w_mod = pad_w % self.stride
            pad_h_mod = pad_h % self.stride
            if pad_w_mod != 0:
                pad_w += (self.stride - pad_w_mod)
            if pad_h_mod != 0:
                pad_h += (self.stride - pad_h_mod)
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        if any([pad_left, pad_top, pad_right, pad_bottom]):
            image = ImageOps.expand(
                image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=self.color)

        # 如果没有 target（推理流程），直接返回 image
        if target is None:
            return image, None

        # update boxes in target (if present)
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].numpy().astype(np.float32)  # [N,4] x1,y1,x2,y2
            # scale coordinates then shift by pad
            boxes = boxes * r
            boxes[:, [0, 2]] += pad_left  # x coords
            boxes[:, [1, 3]] += pad_top   # y coords
            # clip to new image size
            boxes[:, 0] = np.clip(boxes[:, 0], 0, target_w - 1)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, target_h - 1)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, target_w - 1)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, target_h - 1)
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            # update area if exists
            if 'area' in target:
                target['area'] = target['area'] * (r * r)

        return image, target


class Resize:
    """按短边或固定尺寸缩放（保持长宽比），并对 boxes 缩放

    Args:
        min_size: (int | tuple(ints)): 如果是单个 int，将短边缩放到该尺寸（常用于训练时随机选择）；
                  如果是 tuple，代表直接将短边随机选取在该区间内。
        max_size: (int): 缩放后长边不超过 max_size
    """
    def __init__(self, min_size: int = 800, max_size: int = 1333):
        if isinstance(min_size, (list, tuple)):
            self.min_size = min_size
        else:
            self.min_size = (min_size,)
        self.max_size = max_size

    def get_size(self, image_size: Tuple[int, int]):
        w, h = image_size
        min_size = random.choice(self.min_size)
        min_orig = float(min((w, h)))
        max_orig = float(max((w, h)))
        scale = min_size / min_orig
        if max_orig * scale > self.max_size:
            scale = self.max_size / max_orig
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        return new_w, new_h, scale

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None):
        orig_w, orig_h = image.size
        new_w, new_h, scale = self.get_size((orig_w, orig_h))
        image = image.resize((new_w, new_h), resample=Image.BILINEAR) # type: ignore
        if target is None:
            return image, None
        if 'boxes' in target:
            boxes = target['boxes'].numpy()
            boxes = boxes * scale
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            if 'area' in target:
                target['area'] = target['area'] * (scale * scale)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None):
        if random.random() < self.prob:
            image = F.hflip(image) # type: ignore
            w, _ = image.size
            if target is None:
                return image, None
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].numpy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        return image, target


class COCODetectionDataset(Dataset):
    def __init__(
        self,
        root: str,
        ann_file: str,
        transforms: Optional[Callable] = None,
        return_masks: bool = False
    ):
        """
        Args:
            root: images 的根目录
            ann_file: COCO 格式的 annotation json 文件路径
            transforms: callable(image, target) -> image, target
            return_masks: 是否在 target 中返回 segmentation masks（list of polygons/bitmasks）
        """
        if COCO is None:
            raise RuntimeError("""
                pycocotools is required to use COCODetectionDataset.
                Install with 'pip install pycocotools' or 'pip install pycocotools-windows' 
                depending on platform.""")

        self.root = Path(root)
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.return_masks = return_masks

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img_path = self.root / path
        image = Image.open(img_path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []
        for ann in anns:
            # 有些 ann 可能没有 bbox 或者 bbox 无效，需过滤
            if 'bbox' not in ann:
                continue
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                continue
            orig_cat = ann['category_id']
            # 将 COCO 的原始 category_id 映射为 0..79
            if orig_cat not in COCO_ID_TO_80:
                # 如果注释里出现了未在 80 类表中的 id，跳过该注释
                # 这种情况常见于 COCO 的 1..91 id 空洞（非用于检测的 id）
                continue
            mapped_label = COCO_ID_TO_80[orig_cat]
            boxes.append([x, y, x + w, y + h])
            labels.append(mapped_label)
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
            if self.return_masks:
                masks.append(ann.get('segmentation', None))
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8)}
        if self.return_masks:
            target['masks'] = masks # type: ignore

        if self.transforms is not None:
            try:
                image, target = self.transforms(image, target)
            except Exception as e:
                raise RuntimeError(
                    f"Error applying transforms for image id {img_id} ({img_path}). "
                    f"Original error: {e}"
                ) from e

        return image, target


def coco_collate(batch):
    images, targets = zip(*batch)
    # Stack images -> [B, C, H, W]
    inputs = torch.stack(images, dim=0)
    # 计算当前 batch 中最大 bbox 数量
    max_num_boxes = max([len(t['boxes']) for t in targets])
    # 初始化用于存储 padded bboxes 和 labels 的 tensor
    batch_size = len(targets)
    gt_bboxes = torch.zeros((batch_size, max_num_boxes, 4), dtype=torch.float32)
    gt_labels = -torch.ones((batch_size, max_num_boxes), dtype=torch.int64)  # 使用-1填充
    for i, target in enumerate(targets):
        num_boxes = len(target['boxes'])
        if num_boxes > 0:
            gt_bboxes[i, :num_boxes] = target['boxes']
            gt_labels[i, :num_boxes] = target['labels']
    data_samples = {
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels}

    return inputs, data_samples


def get_train_transforms(
    imgsz: int = 640,
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    auto = True
):
    transforms = Compose([
        LetterBox(img_size=imgsz, stride=32, auto=auto, scaleup=True, color=114),
        RandomHorizontalFlip(0.5),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    return transforms

def get_val_transforms(
    imgsz: int = 640,
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    auto = True
):
    transforms = Compose([
        LetterBox(img_size=imgsz, stride=32, auto=auto, scaleup=False, color=114),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    return transforms


def get_infer_transforms(
    imgsz: int = 640,
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225),
    auto = True
):
    """
    推理时使用的 transforms：
      - LetterBox(scaleup=False) 保持长宽比并 pad 到 imgsz（同 val）
      - ToTensor
      - Normalize

    返回的是一个 callable (image, target=None) -> (image_tensor, None)
    """
    transforms = InferCompose([
        LetterBox(img_size=imgsz, stride=32, auto=auto, scaleup=False, color=114),
        ToTensor(),
        Normalize(mean=mean, std=std)])

    return transforms


def infer_collate(batch):
    """
    batch: list of items each returned by dataset.__getitem__:
        - if dataset.__getitem__ returns (image_tensor, None) -> batch is list of tuples
    返回:
        inputs: Tensor [B, C, H, W]
        meta: list of meta info or None (这里不包含 target)
    """
    images = [item[0] if isinstance(item, (list, tuple)) else item for item in batch]
    inputs = torch.stack(images, dim=0)

    return inputs
