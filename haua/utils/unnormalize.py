from PIL import Image

import torch
import torchvision.transforms as transforms


def tensor2image(
    norm_tensor,
    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225],
    show = False,
    save_path = None
):
    """
    将经过归一化的图像Tensor转换为PIL图像并展示。

    参数:
        tensor (torch.Tensor): 经过归一化的图像Tensor。
        mean (list): 归一化时使用的均值。
        std (list): 归一化时使用的标准差。
    """
    unnormalize = transforms.Normalize(
        mean = [-m/s for m, s in zip(mean, std)],
        std = [1/s for s in std])
    # 应用反归一化变换
    unnormalized_tensor = unnormalize(norm_tensor)
    # 将Tensor的值限制在[0, 1]范围内
    unnormalized_tensor = torch.clamp(unnormalized_tensor, 0, 1)
    # 将Tensor转换为PIL图像
    to_pil = transforms.ToPILImage()
    image = to_pil(unnormalized_tensor)
    # 展示图像
    if show:
        image.show()
    if save_path:
        image.save(save_path)

    return image