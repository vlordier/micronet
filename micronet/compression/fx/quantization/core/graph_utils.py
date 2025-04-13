# quantization_framework/quant_core/graph_utils.py
import sys
from typing import Type, Callable, Tuple, Set
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入 F 便于使用

# --- 常量定义 ---

# 定义通常需要量化权重的模块类型
DEFAULT_WEIGHT_QUANT_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    # --- 标准层 ---
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    # --- 转置卷积 ---
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    # --- 嵌入层 (通常量化) ---
    nn.Embedding,
    nn.EmbeddingBag,
    # --- 循环层 (注意：内部线性层的权重是量化目标) ---
    # 直接量化整个 nn.LSTM/GRU 比较复杂，通常是量化其内部的 gate/projection 线性层
    # 如果你的框架支持深入模块内部并替换，可以考虑添加
    nn.LSTM,  # 通常需要特殊处理
    nn.GRU,  # 通常需要特殊处理
    # --- 动态量化中可能考虑的模块 ---
    nn.LSTMCell,  # 同上
    nn.GRUCell,  # 同上
)

# 定义通常需要量化其输出激活的模块类型
# 注意：这通常发生在不进行算子融合的情况下，或者某些特定模块的输出语义上需要观察
DEFAULT_ACTIVATION_QUANT_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    # --- 计算密集型层 (无融合时) ---
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Linear,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    # --- 归一化层 (无融合时) ---
    nn.BatchNorm1d,  # 通常融合到前面的卷积/线性层
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    # --- 池化层 (输出有时需要观察) ---
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    # MaxPool 通常不量化其输出，但可以观察
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    # --- 嵌入层 (输出通常被观察) ---
    nn.Embedding,
    nn.EmbeddingBag,
    # --- 激活函数模块 (如果用模块形式且无融合) ---
    nn.ReLU,
    nn.ReLU6,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardswish,
    nn.SiLU,
    nn.GELU,
)

# 定义通常需要量化其输出激活的 torch 函数或 operator
# (核心量化目标)
DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES: Set[Callable] = {
    # --- 逐元素算术操作 ---
    torch.add,
    operator.add,
    torch.mul,
    operator.mul,
    torch.sub,
    operator.sub,
    torch.div,
    operator.truediv,
    # --- 激活函数 ---
    torch.relu,
    F.relu,
    F.relu6,
    torch.sigmoid,
    torch.special.expit,
    F.sigmoid,  # sigmoid 有多种形式
    torch.tanh,
    F.tanh,
    F.hardswish,  # nn.Hardswish 的函数形式
    F.silu,  # nn.SiLU 的函数形式 (Swish)
    F.gelu,  # nn.GELU 的函数形式
    F.leaky_relu,
    # --- 池化函数 ---
    F.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_avg_pool3d,
    F.avg_pool1d,
    F.avg_pool2d,
    F.avg_pool3d,
    F.max_pool1d,  # MaxPool 通常不量化输出
    F.max_pool2d,
    F.max_pool3d,
    # --- 归一化函数 (无融合时) ---
    F.batch_norm,  # 通常融合
    F.layer_norm,
    F.instance_norm,
    # --- 矩阵/向量操作 ---
    torch.matmul,  # 矩阵乘法
    torch.bmm,  # 批量矩阵乘法
    # --- 拼接/分割/堆叠 ---
    torch.cat,  # 连接操作通常需要量化输出
    torch.stack,  # 堆叠操作
    # --- 其他可能需要量化的函数 ---
    torch.mean,  # 取平均值
    F.interpolate,  # 插值操作
    # --- 注意力相关 (示例) ---
    F.softmax,  # Softmax 输出通常在 [0, 1]，有时不需要再量化，但可以观察
    F.scaled_dot_product_attention,  # 复杂操作，可能需要特殊处理
}

# 定义可能需要量化其输出的 Tensor 方法名称
DEFAULT_ACTIVATION_QUANT_METHOD_NAMES: Tuple[str, ...] = (
    # --- 逐元素算术操作 ---
    "add",  # a.add(b)
    "mul",  # a.mul(b)
    "sub",  # a.sub(b) (有时)
    "div",  # a.div(b) (有时)
    "__add__",  # a + b (会被追踪为 call_method)
    "__mul__",  # a * b
    "__sub__",  # a - b
    "__truediv__",  # a / b
    # --- 激活函数 ---
    "relu",
    "relu_",  # inplace relu
    "sigmoid",
    "sigmoid_",
    "tanh",
    "tanh_",
    # --- 其他常用方法 ---
    "mean",  # tensor.mean()
    "clamp",  # tensor.clamp() (限制范围，有时用于模拟量化)
    # "view", # 形状变换通常不量化
    # "reshape",
    # "permute",
    # "transpose",
    # "contiguous",
)

# --- 工具函数 (保持不变) ---


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


# --- 颜色和打印工具 (保持不变) ---
# ANSI 颜色代码
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"

# 预定义颜色用途
COLOR_DEBUG = COLOR_CYAN
COLOR_INFO = COLOR_RESET
COLOR_WARN = COLOR_YELLOW
COLOR_ERROR = COLOR_RED
COLOR_SUCCESS = COLOR_GREEN
COLOR_MODULE = COLOR_MAGENTA
COLOR_OPERATOR = COLOR_BLUE
COLOR_NODE = COLOR_BOLD + COLOR_YELLOW

_use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, color_code: str) -> str:
    """如果支持，给文本添加 ANSI 颜色代码"""
    return f"{color_code}{text}{COLOR_RESET}" if _use_color else text
