# quantization_framework/quant_core/graph_utils.py
from typing import Callable

import torch
import torch.nn as nn

# 定义可量化权重的模块类型 (可以根据需要扩展)
DEFAULT_WEIGHT_QUANT_MODULE_TYPES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)

# 定义其输出激活通常需要量化的模块类型 (可以根据需要扩展)
DEFAULT_ACTIVATION_QUANT_MODULE_TYPES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ReLU,  # 示例：也可能想量化 ReLU 的输出
    nn.MaxPool2d,  # 等等
)

# 定义可能需要量化其输出的 torch 函数 (示例)

DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES = (
    torch.add,
    torch.cat,
    torch.mul,
)


def is_quantizable_weight_module(module: nn.Module) -> bool:
    """检查一个模块是否是通常需要量化权重的类型"""
    return isinstance(module, DEFAULT_WEIGHT_QUANT_MODULE_TYPES)


def is_quantizable_activation_module(module: nn.Module) -> bool:
    """检查一个模块类型，其输出激活通常需要量化"""
    return isinstance(module, DEFAULT_ACTIVATION_QUANT_MODULE_TYPES)


def is_quantizable_activation_function(func: Callable) -> bool:
    """检查一个 torch 函数是否是其输出激活通常需要量化的类型"""
    return func in DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES
