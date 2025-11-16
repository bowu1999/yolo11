def make_divisible(x, divisor=8):
    """将通道数调整为8的倍数"""
    return int((x + divisor / 2) // divisor * divisor)