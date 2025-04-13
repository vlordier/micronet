import sys
from typing import Type, Callable, Tuple, Set
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 常量定义 ---

# 定义通常需要量化权重的模块类型
DEFAULT_WEIGHT_QUANT_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Embedding,
    nn.EmbeddingBag,
    nn.LSTM,
    nn.GRU,  # 通常需要特殊处理内部层
    nn.LSTMCell,
    nn.GRUCell,
)

# 定义通常需要量化其输出激活的模块类型
DEFAULT_ACTIVATION_QUANT_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,  # 通常融合
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,  # 输出通常不量化，但可观察
    nn.Embedding,
    nn.EmbeddingBag,
    nn.ReLU,
    nn.ReLU6,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardswish,
    nn.SiLU,
    nn.GELU,
)

# 定义通常需要量化其输出激活的 torch 函数或 operator
DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES: Set[Callable] = {
    torch.add,
    operator.add,
    torch.mul,
    operator.mul,
    torch.sub,
    operator.sub,
    torch.div,
    operator.truediv,
    torch.relu,
    F.relu,
    F.relu6,
    torch.sigmoid,
    torch.special.expit,
    F.sigmoid,
    torch.tanh,
    F.tanh,
    F.hardswish,
    F.silu,
    F.gelu,
    F.leaky_relu,
    F.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_avg_pool3d,
    F.avg_pool1d,
    F.avg_pool2d,
    F.avg_pool3d,
    F.max_pool1d,
    F.max_pool2d,
    F.max_pool3d,  # 输出通常不量化
    F.batch_norm,
    F.layer_norm,
    F.instance_norm,  # 通常融合
    torch.matmul,
    torch.bmm,
    torch.cat,
    torch.stack,
    torch.mean,
    F.interpolate,
    F.softmax,
    F.scaled_dot_product_attention,  # 可能需要特殊处理
}

# 定义可能需要量化其输出的 Tensor 方法名称
DEFAULT_ACTIVATION_QUANT_METHOD_NAMES: Tuple[str, ...] = (
    "add",
    "mul",
    "sub",
    "div",
    "__add__",
    "__mul__",
    "__sub__",
    "__truediv__",
    "relu",
    "relu_",
    "sigmoid",
    "sigmoid_",
    "tanh",
    "tanh_",
    "mean",
    "clamp",
)

# --- 工具函数 ---


def is_quantizable_weight_module(module: nn.Module) -> bool:
    """检查一个模块是否是通常需要量化权重的类型"""
    return isinstance(module, DEFAULT_WEIGHT_QUANT_MODULE_TYPES)


def is_quantizable_activation_module(module: nn.Module) -> bool:
    """检查一个模块类型，其输出激活通常需要量化"""
    return isinstance(module, DEFAULT_ACTIVATION_QUANT_MODULE_TYPES)


def is_quantizable_activation_function(func: Callable) -> bool:
    """检查一个 torch 函数或 operator 是否是其输出激活通常需要量化的类型"""
    return func in DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES


def is_quantizable_activation_method(method_name: str) -> bool:
    """检查一个方法名称是否是其输出激活通常需要量化的类型"""
    return method_name in DEFAULT_ACTIVATION_QUANT_METHOD_NAMES


# --- 颜色和打印工具 ---
# ANSI 颜色代码
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_WHITE = "\033[97m"  # 添加白色

# --- 基础语义颜色 ---
COLOR_DEBUG = COLOR_CYAN  # 用于详细的调试步骤
COLOR_INFO = COLOR_RESET  # 用于常规流程信息 (默认终端颜色)
COLOR_WARN = COLOR_YELLOW  # 用于警告或可跳过的问题
COLOR_ERROR = COLOR_RED  # 用于错误或失败
COLOR_SUCCESS = COLOR_GREEN  # 用于成功完成的操作

# --- FX 图元素颜色 ---
COLOR_MODULE = COLOR_MAGENTA  # 用于模块名称或模块调用操作
COLOR_OPERATOR = COLOR_BLUE  # 用于函数/方法/内置操作符名称
COLOR_NODE = COLOR_BOLD + COLOR_WHITE  # 用于 FX 节点名称 (加粗白色更通用)
COLOR_TARGET = COLOR_CYAN  # 用于节点的目标 (属性, 权重/激活标识)
COLOR_INPUT = COLOR_BOLD + COLOR_GREEN  # 用于输入/占位符节点

# --- 量化流程语义颜色 ---
COLOR_PHASE = COLOR_BOLD + COLOR_MAGENTA  # 用于标记主要阶段 (Prepare/Convert)
COLOR_ACTION = COLOR_BOLD + COLOR_BLUE  # 用于标记具体操作 (INSERT/REMOVE/REPLACE)
COLOR_REASON = COLOR_YELLOW  # 用于标记跳过或特定决策的原因 (与WARN类似)

_use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, color_code: str) -> str:
    """如果支持，给文本添加 ANSI 颜色代码"""
    # 确保总是以 RESET 结尾，即使 color_code 包含 BOLD 等
    # BOLD 等修饰符本身不带颜色，需要和具体颜色组合
    # 如果 color_code 已经是类似 COLOR_BOLD + COLOR_RED 的组合，它会正常工作
    # 如果 color_code 只是 COLOR_BOLD，这里 text 会变粗体，然后 reset
    return f"{color_code}{text}{COLOR_RESET}" if _use_color else text
