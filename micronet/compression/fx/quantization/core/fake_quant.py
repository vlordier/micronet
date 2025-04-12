# quantization_framework/quant_core/fake_quant.py
import torch.nn as nn


class PlaceholderFakeQuant(nn.Module):
    """
    一个简单的占位符 FakeQuant，用于标记需要插入伪量化节点的位置。
    在 QAT 准备阶段插入，在训练时模拟量化效应。
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # 同上，可以存储元数据
        self.placeholder_args = args
        self.placeholder_kwargs = kwargs
        # 对于 QAT，这里的参数（scale/zero_point）最终需要是可学习的
        # 但在这个占位符阶段，我们还不添加它们

    def forward(self, x):
        # 在准备阶段，它只是一个标识符
        return x

    def __repr__(self):
        return (
            f"PlaceholderFakeQuant({self.placeholder_args}, {self.placeholder_kwargs})"
        )


# 未来可以定义 LearnableFakeQuant 等
