# quantization_framework/quant_core/observer.py
import torch.nn as nn


class PlaceholderObserver(nn.Module):
    """
    一个简单的占位符 Observer，用于标记需要观察激活或权重的位置。
    在 PTQ 校准阶段，它将被替换或填充统计数据。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # 这里可以存储一些元数据，比如数据类型、量化方案等，但现在保持简单
        self.placeholder_args = args
        self.placeholder_kwargs = kwargs

    def forward(self, x):
        # 在准备阶段，它只是一个标识符，不做任何事情
        return x

    def __repr__(self):
        return (
            f"PlaceholderObserver({self.placeholder_args}, {self.placeholder_kwargs})"
        )


# 可以根据需要定义更具体的占位符，比如 MinMaxObserverPlaceholder 等
