from typing import Callable, Optional, Any, Dict, Type, Union
import torch
import torch.nn as nn

from ..utils import get_script_name


_SCRIPT_NAME = get_script_name(__file__)

class BaseModel(nn.Module):
    """
    通用模型基类，支持以下两种模块定义方式：
        - 直接传入实例化的 nn.Module
        - 传入模块类（未实例化）+ module_configs 自动实例化
        【注】实例优先：如果传入的是实例，直接使用，忽略 config
    """

    def __init__(
        self,
        backbone: Union[nn.Module, Type[nn.Module]],
        neck: Optional[Union[nn.Module, Type[nn.Module]]] = None,
        head: Optional[Union[nn.Module, Type[nn.Module]]] = None,
        custom_postprocess: Optional[Callable[[Any], Any]] = None,
        module_configs: Optional[Dict[str, dict]] = None,
    ):
        super().__init__()
        self.module_configs = module_configs or {}
        self.backbone = self._init_module("backbone", backbone)
        self.neck = self._init_module("neck", neck) if neck else nn.Identity()
        self.head = self._init_module("head", head) if head else nn.Identity()
        self.custom_postprocess = custom_postprocess

    def _init_module(
        self,
        name: str,
        module_or_class: Union[nn.Module, Type[nn.Module]]
    ) -> nn.Module:
        """初始化模块：实例直接返回；类则用 config 实例化"""
        if isinstance(module_or_class, nn.Module):
            return module_or_class
        elif isinstance(module_or_class, type) and issubclass(module_or_class, nn.Module):
            cfg = self.module_configs.get(name, {})
            return module_or_class(**cfg)
        else:
            raise TypeError(
                f"{name} 必须是 nn.Module 实例 或 nn.Module 子类，当前类型为 {type(module_or_class)}")

    def forward_backbone(self, x: torch.Tensor) -> Any:
        return self.backbone(x)

    def forward_neck(self, feats: Any) -> Any:
        return self.neck(feats)

    def forward_head(self, feats: Any) -> Any:
        return self.head(feats)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        feats = self.forward_backbone(x)
        # print(f"[{_SCRIPT_NAME} - {self.class_name} - forward] feats: ", [f.shape for f in feats])
        fused = self.forward_neck(feats)
        out = self.forward_head(fused)
        if self.custom_postprocess is not None:
            out = self.custom_postprocess(out)

        return feats, fused, out

    def summary(self) -> Dict[str, str]:
        return {
            "backbone": self.backbone.__class__.__name__,
            "neck": self.neck.__class__.__name__,
            "head": self.head.__class__.__name__,
            "has_custom_postprocess": str(self.custom_postprocess is not None)}

    def set_postprocess(self, fn: Callable[[Any], Any]):
        self.custom_postprocess = fn

    def fuse(self, verbose: bool = True) -> "BaseModel":
        """自动 fuse 模型中所有支持 .fuse() 的子模块（如 ConvBNAct）"""
        fused_count = 0

        def _fuse_recursive(module: nn.Module) -> int:
            count = 0
            if hasattr(module, 'fuse') and callable(getattr(module, 'fuse')):
                if hasattr(module, 'bn'):  # 针对 ConvBNAct 类
                    module.fuse()
                    count += 1
            for child in module.children():
                count += _fuse_recursive(child)

            return count

        for name in ["backbone", "neck", "head"]:
            module = getattr(self, name, None)
            if module is not None:
                fused_count += _fuse_recursive(module)
        if verbose:
            print(f"BaseModel.fuse(): 成功融合 {fused_count} 个可融合模块。")

        return self
    
    @property
    def class_name(self):
        return self.__class__.__name__