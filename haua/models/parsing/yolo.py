from typing import Any, Union, Tuple

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from ..utils import make_grid


class YOLODecoder():
    def __init__(
        self,
        threshold = 0.5,
        strides = [8, 16, 32],
        img_size = 640,
        reg_max = 16,
        cls_num = 80,
        prob_fn = 'softmax',
        padding = True
    ):
        if isinstance(img_size, tuple):
            self.img_size = img_size
        elif isinstance(img_size, int):
            self.img_size = (img_size,) * 2
        self.strides = strides
        self.scales = [self.img_size[0] // i for i in strides]
        self.threshold = threshold
        self.reg_max = reg_max
        self.cls_num = cls_num
        self.channel = 4 * reg_max + cls_num
        self.prob_fn = prob_fn
        self.padding = padding

    def __call__(self, outputs, original_img_size=None):
        if isinstance(outputs[-1], tuple):
            outputs = outputs[-1]

        all_results = []
        all_grids = []
        all_strides = []

        for i in range(len(self.strides)):
            _, _, H, W = outputs[i].shape
            N = H * W
            all_results.append(outputs[i].permute(0, 2, 3, 1).view(-1, self.channel))
            all_grids.append(make_grid((self.scales[i], ) * 2).view(-1, 2))
            all_strides.append(torch.tensor([self.strides[i]] * N))

        all_results = torch.cat(all_results, dim=0)
        all_grids = torch.cat(all_grids, dim=0)
        all_strides = torch.cat(all_strides, dim=0)

        dfl_result, cls_result = torch.split(
            all_results, split_size_or_sections=[4 * self.reg_max, self.cls_num], dim=1)

        if self.prob_fn == 'softmax':
            cls_result = F.softmax(cls_result, dim=1)
        elif self.prob_fn == 'sigmoid':
            cls_result = F.sigmoid(cls_result)

        cls_, indices = torch.max(cls_result, dim=1)
        resindx = cls_ > self.threshold

        dfl_res = dfl_result[resindx]
        grids_res = all_grids[resindx]
        strides_res = all_strides[resindx]
        cls_res = indices[resindx]
        score_res = cls_[resindx]
        bbox_res = self._decode_dfl(dfl_res, grids_res, strides_res)

        if self.padding and original_img_size is not None:
            input_w, input_h = self.img_size
            orig_w, orig_h = original_img_size

            # 计算 letterbox 缩放比例
            scale = min(input_w / orig_w, input_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            pad_left = (input_w - new_w) // 2
            pad_top = (input_h - new_h) // 2

            # 1. 减去 pad
            bbox_res[:, [0, 2]] -= pad_left
            bbox_res[:, [1, 3]] -= pad_top

            # 2. 除以 scale
            bbox_res[:, [0, 2]] /= scale
            bbox_res[:, [1, 3]] /= scale

            # 3. Clamp 到原始图像范围
            bbox_res[:, [0, 2]] = torch.clamp(bbox_res[:, [0, 2]], 0, orig_w)
            bbox_res[:, [1, 3]] = torch.clamp(bbox_res[:, [1, 3]], 0, orig_h)

        return cls_res, score_res, bbox_res

    def _decode_dfl(self, dfl_logits, grid, strides):
        assert dfl_logits.shape[1] == 4 * (self.reg_max), \
            f"Expected {4 * (self.reg_max)} channels, got {dfl_logits.shape[1]}"
        assert grid.shape[1] == 2, "Grid must be (N, 2)"
        assert strides.shape[0] == dfl_logits.shape[0], "Strides must have same length as dfl_logits"
        N = dfl_logits.shape[0]

        dfl_reshaped = dfl_logits.view(N, 4, self.reg_max)
        dfl_probs = dfl_reshaped.softmax(dim=-1)
        discrete_values = torch.arange(self.reg_max, dtype=dfl_logits.dtype, device=dfl_logits.device)
        offsets = (dfl_probs * discrete_values).sum(dim=-1)

        strides_expanded = strides.unsqueeze(-1).expand_as(offsets)
        offsets_scaled = offsets * strides_expanded
        l, t, r, b = offsets_scaled[:, 0], offsets_scaled[:, 1], offsets_scaled[:, 2], offsets_scaled[:, 3]
        cx = grid[:, 0] * strides
        cy = grid[:, 1] * strides

        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b
        bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

        return bboxes


class YoloResult:
    def __init__(
        self,
        image,
        cls_res,
        score_res,
        bbox_res,
        class_names,
        conf_threshold=0.0,
        target_classes=None
    ):
        # 处理图像输入
        if isinstance(image, str) or isinstance(image, Path):
            self.image = cv2.imread(str(image))
            if self.image is None:
                raise ValueError(f"无法读取图像: {image}")
        elif isinstance(image, np.ndarray):
            self.image = image.copy()
            if self.image.ndim == 3 and self.image.shape[2] == 3:
                pass
            else:
                raise ValueError("图像必须是 HxWx3 的 NumPy 数组")
        else:
            raise TypeError("image 必须是文件路径或 NumPy 数组")

        self.cls_res = self._to_numpy(cls_res)
        self.score_res = self._to_numpy(score_res)
        self.bbox_res = self._to_numpy(bbox_res)

        assert len(self.cls_res) == len(self.score_res) == len(self.bbox_res)

        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.target_classes = set(target_classes) if target_classes is not None else None

        self.color_map = self._get_color_map()

        # 过滤结果
        self._filter_detections()

    def _to_numpy(self, x):
        if hasattr(x, 'cpu'):
            return x.cpu().detach().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    def _filter_detections(self):
        keep = self.score_res >= self.conf_threshold
        if self.target_classes is not None:
            in_target = np.array([c in self.target_classes for c in self.cls_res])
            keep = keep & in_target
        idx = np.where(keep)[0]
        self.cls_res = self.cls_res[idx]
        self.score_res = self.score_res[idx]
        self.bbox_res = self.bbox_res[idx]

    def _get_color_map(self):
        """为每个类别生成固定、稳定、不会重复的 BGR 颜色"""
        color_map = {}

        for i, name in enumerate(self.class_names):
            # 使用名称做哈希，稳定
            seed = abs(hash(name)) % (2**32)
            rng = np.random.default_rng(seed)

            # 在 HSV 空间随机生成，但种子稳定 → 颜色固定
            h = rng.uniform(0, 1)  # hue
            s = rng.uniform(0.6, 1.0)
            v = rng.uniform(0.7, 1.0)

            hsv = np.array([[[h * 179, s * 255, v * 255]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]

            color_map[i] = tuple(int(x) for x in bgr)

        return color_map

    def _cls_color(self, cls_id):
        """返回某个类别的固定 BGR 颜色"""
        return self.color_map.get(cls_id, (0, 255, 0))

    @property
    def show(self):
        def _show(figsize=(12, 8), title="YOLO Detection"):
            img_draw = self.image.copy()

            for cls_id, score, (x1, y1, x2, y2) in zip(
                self.cls_res, self.score_res, self.bbox_res
            ):
                cls_id = int(cls_id)
                label = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class{cls_id}"
                text = f"{label} {score:.2f}"

                color = self._cls_color(cls_id)  # ✨ 获取固定颜色

                # 边框
                cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # 标签背景
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(
                    img_draw,
                    (int(x1), int(y1) - 20),
                    (int(x1) + w, int(y1)),
                    color,
                    -1,
                )

                # 文本
                cv2.putText(
                    img_draw,
                    text,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            img_rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=figsize)
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return _show

    def __call__(self):
        self.show()
