import torch


def make_grid(h: int, w: int, stride: int = 1, device=None, dtype=torch.float32):
    """
    创建 feature map 的格点中心坐标 grid
    Args:
        h, w (int):
            特征图的高度与宽度

        stride (int):
            特征图相对于原图的下采样倍数（一个格子对应多少像素）

        device, dtype:
            返回 Tensor 的设备与数据类型

    Returns:
        grid (Tensor):
            形状 (H, W, 2) 的网格，每个位置为 (cx, cy)
    """
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1)

    return grid
