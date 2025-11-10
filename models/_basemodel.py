from typing import Callable, Optional, Any, Dict, Type, Union
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    通用模型基类
    支持以下三种模块定义方式：
        直接传入实例化的 nn.Module
        传入模块类（未实例化）
        传入模块类 + module_configs（自动实例化）
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Type[nn.Module]],
        neck: Union[nn.Module, Type[nn.Module]],
        head: Union[nn.Module, Type[nn.Module]],
        custom_postprocess: Optional[Callable[[Any], Any]] = None,
        module_configs: Optional[Dict[str, dict]] = None,
    ):
        """
        Args:
            backbone: 特征提取网络结构，可以是实例或类
            neck: 特征融合网络结构，可以是实例或类
            head: 任务预测网络结构，可以是实例或类
            custom_postprocess: 自定义输出后处理函数
            module_configs: 模块初始化参数字典，如：
                {
                    "backbone": {"depth": 21, "in_channels": 3},
                    "neck": {"in_channels": [128, 256, 512]},
                    "head": {"num_classes": 80}}
        """
        super().__init__()
        self.module_configs = module_configs or {}
        self.backbone = self._init_module("backbone", backbone)
        self.neck = self._init_module("neck", neck)
        self.head = self._init_module("head", head)
        self.custom_postprocess = custom_postprocess

    def _init_module(
        self,
        name: str,
        module_or_class: Union[nn.Module, Type[nn.Module]]
    ) -> nn.Module:
        """判断输入是否为类或实例，根据情况实例化模块。"""
        cfg = self.module_configs.get(name, {})
        if isinstance(module_or_class, nn.Module):
            if cfg: # 若为实例，可重新根据配置参数重新实例化
                cls = module_or_class.__class__
                return cls(**cfg)
            return module_or_class
        elif isinstance(module_or_class, type) and issubclass(module_or_class, nn.Module):
            return module_or_class(**cfg) # 若为类，则用配置参数实例化
        else:
            raise TypeError(
                f"{name} 必须是 nn.Module 实例 或 nn.Module 类，当前类型为 {type(module_or_class)}")

    def forward_backbone(self, x: torch.Tensor) -> Any:
        return self.backbone(x)

    def forward_neck(self, feats: Any) -> Any:
        return self.neck(feats)

    def forward_head(self, feats: Any) -> Any:
        return self.head(feats)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        feats = self.forward_backbone(x)
        fused = self.forward_neck(feats)
        out = self.forward_head(fused)
        if self.custom_postprocess is not None:
            out = self.custom_postprocess(out)

        return out

    def summary(self) -> Dict[str, str]:
        """模型结构摘要"""
        return {
            "backbone": self.backbone.__class__.__name__,
            "neck": self.neck.__class__.__name__,
            "head": self.head.__class__.__name__,
            "has_custom_postprocess": str(self.custom_postprocess is not None)}

    def set_postprocess(self, fn: Callable[[Any], Any]):
        """动态设置后处理函数"""
        self.custom_postprocess = fn
