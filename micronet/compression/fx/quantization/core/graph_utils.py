import sys
from typing import List, Dict, Tuple, Type, Optional, Set, Callable
import operator
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx
from torch.fx.graph_module import GraphModule


# --- 日志记录器设置 ---
graph_utils_logger = logging.getLogger("micronet.fx.quantizer.graph_utils")
# 如果没有配置，添加默认 handler
if not graph_utils_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    graph_utils_logger.addHandler(handler)
    # Quantizer 初始化时会根据 debug 参数调整
    graph_utils_logger.setLevel(logging.INFO)
    graph_utils_logger.propagate = False


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
    nn.GRU,
    nn.LSTMCell,
    nn.GRUCell,
    nn.BatchNorm1d,  # 通常融合
    nn.BatchNorm2d,  # 通常融合
    nn.BatchNorm3d,  # 通常融合
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
    nn.BatchNorm1d,  # 通常融合
    nn.BatchNorm2d,  # 通常融合
    nn.BatchNorm3d,  # 通常融合
    nn.LayerNorm,
    nn.InstanceNorm1d,  # 通常不融合
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
    nn.MaxPool3d,
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
    torch.special.expit,  # 等价于 sigmoid
    F.sigmoid,
    torch.tanh,
    F.tanh,
    F.hardswish,
    F.silu,  # 等价于 Swish
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
    F.max_pool3d,
    F.batch_norm,  # 通常融合
    F.layer_norm,
    F.instance_norm,
    torch.matmul,
    torch.bmm,
    torch.cat,
    torch.stack,
    torch.mean,
    F.interpolate,
    F.softmax,
    F.scaled_dot_product_attention,
}

# 定义通常需要量化其输出的 Tensor 方法名称
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

# --- 类型检查工具函数 ---


def is_quantizable_weight_module(module: nn.Module) -> bool:
    """检查一个模块是否是通常需要量化权重的类型"""
    return isinstance(module, DEFAULT_WEIGHT_QUANT_MODULE_TYPES)


def is_quantizable_activation_module(module: nn.Module) -> bool:
    """检查一个模块是否是通常需要量化其输出激活的类型"""
    # 经过融合后，BN 层就不在这里判断了
    return isinstance(module, DEFAULT_ACTIVATION_QUANT_MODULE_TYPES)


def is_quantizable_activation_function(func: Callable) -> bool:
    """检查一个 torch 函数或 operator 是否是通常需要量化其输出激活的类型"""
    return func in DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES


def is_quantizable_activation_method(method_name: str) -> bool:
    """检查一个方法名称是否是通常需要量化其输出激活的类型"""
    return method_name in DEFAULT_ACTIVATION_QUANT_METHOD_NAMES


# --- 颜色和打印工具 ---
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_MAGENTA = "\033[95m"
COLOR_CYAN = "\033[96m"
COLOR_WHITE = "\033[97m"
COLOR_DEBUG = COLOR_CYAN
COLOR_INFO = COLOR_RESET
COLOR_WARN = COLOR_YELLOW
COLOR_ERROR = COLOR_RED
COLOR_SUCCESS = COLOR_GREEN
COLOR_MODULE = COLOR_MAGENTA
COLOR_OPERATOR = COLOR_BLUE
COLOR_NODE = COLOR_BOLD + COLOR_WHITE
COLOR_TARGET = COLOR_CYAN
COLOR_INPUT = COLOR_BOLD + COLOR_GREEN
COLOR_PHASE = COLOR_BOLD + COLOR_MAGENTA
COLOR_ACTION = COLOR_BOLD + COLOR_BLUE
COLOR_REASON = COLOR_YELLOW

_use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _colorize(text: str, color_code: str) -> str:
    """如果支持，给文本添加 ANSI 颜色代码"""
    return f"{color_code}{text}{COLOR_RESET}" if _use_color else text


# --- 用于获取/删除嵌套模块的辅助函数 ---


def _get_nested_module(model: nn.Module, target: str) -> Optional[nn.Module]:
    """
    根据点分隔的目标字符串从模型中获取嵌套模块。
    例如 'layer1.0.conv1'
    """
    current = model
    try:
        for name in target.split("."):
            current = getattr(current, name)
        return current
    except AttributeError:
        graph_utils_logger.error(f"无法在模型中找到模块: {target}")
        return None


def _delete_module(model: nn.Module, target: str):
    """
    根据点分隔的目标字符串从模型中删除嵌套模块。
    """
    parts = target.split(".")
    if len(parts) == 1:
        try:
            delattr(model, parts[0])
        except AttributeError:
            graph_utils_logger.warning(f"尝试删除不存在的顶层模块: {target}")
    else:
        parent_target = ".".join(parts[:-1])
        child_name = parts[-1]
        parent_module = _get_nested_module(model, parent_target)
        if parent_module:
            try:
                delattr(parent_module, child_name)
            except AttributeError:
                graph_utils_logger.warning(
                    f"尝试从父模块 '{parent_target}' 删除不存在的子模块: {child_name}"
                )
        else:
            graph_utils_logger.warning(
                f"无法找到父模块 '{parent_target}' 来删除子模块 '{child_name}'"
            )


# --- Conv/Linear + BN 融合函数 ---


def fuse_conv_linear_bn_fx(
    model: GraphModule, modules_to_fuse: List[List[str]]
) -> GraphModule:
    """
    在 FX GraphModule 中手动融合指定的 Conv/Linear 和 BatchNorm 层。

    此函数执行以下操作：
    1. 计算融合后的权重和偏置。
    2. 将融合后的参数更新到 Conv/Linear 模块。
    3. 修改 FX 图：将所有 BN 节点的输出用户重定向到 Conv/Linear 节点，然后删除 BN 节点。
    4. 从模型中删除原始的 BatchNorm 模块实例。
    5. 清理图并重新编译 GraphModule。

    Args:
        model (GraphModule): 要进行融合的 FX GraphModule。**此对象将被就地修改**。
        modules_to_fuse (List[List[str]]): 一个列表，每个元素是一个包含两个字符串的列表，
                                            表示要融合的 [Conv/Linear 模块名称, BatchNorm 模块名称]。

    Returns:
        GraphModule: 修改后的 GraphModule (与输入是同一个对象，但已被修改)。
    """
    if not modules_to_fuse:
        graph_utils_logger.info("没有需要融合的模块对。")
        return model

    graph_utils_logger.info(f"开始 Conv/Linear-BN 融合 ({len(modules_to_fuse)} 对)...")
    model.eval()  # 融合需要 BN 运行统计数据，确保在 eval 模式

    # 需要完整模块字典以通过名称获取模块
    module_dict = dict(model.named_modules())
    nodes_by_target: Dict[str, fx.Node] = {
        str(node.target): node for node in model.graph.nodes if node.op == "call_module"
    }

    fused_count = 0
    for conv_or_linear_name, bn_name in modules_to_fuse:
        graph_utils_logger.debug(
            f"  尝试融合: {_colorize(conv_or_linear_name, COLOR_MODULE)} + {_colorize(bn_name, COLOR_MODULE)}"
        )

        # 1. 获取模块实例
        conv_or_linear_module = module_dict.get(conv_or_linear_name)
        bn_module = module_dict.get(bn_name)

        if not conv_or_linear_module or not bn_module:
            graph_utils_logger.warning(
                f"    跳过：找不到模块 {conv_or_linear_name} 或 {bn_name}。可能已被先前步骤处理或名称错误。"
            )
            continue

        # 检查模块类型
        supported_convs = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        supported_linears = (nn.Linear,)
        supported_bns = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

        if not isinstance(conv_or_linear_module, (supported_convs + supported_linears)):
            graph_utils_logger.warning(
                f"    跳过：模块 {conv_or_linear_name} ({type(conv_or_linear_module).__name__}) 不是支持的 Conv/Linear 类型。"
            )
            continue
        if not isinstance(bn_module, supported_bns):
            graph_utils_logger.warning(
                f"    跳过：模块 {bn_name} ({type(bn_module).__name__}) 不是支持的 BatchNorm 类型。"
            )
            continue

        # 检查 BN 是否有运行统计数据
        if (
            not hasattr(bn_module, "running_mean")
            or not hasattr(bn_module, "running_var")
            or bn_module.running_mean is None
            or bn_module.running_var is None
        ):
            graph_utils_logger.warning(
                f"    跳过：BatchNorm 模块 {bn_name} 缺少运行统计数据 (running_mean/running_var)。请确保模型已训练并在 eval() 模式。"
            )
            continue

        # 2. 获取参数
        w_conv = conv_or_linear_module.weight
        b_conv = conv_or_linear_module.bias

        running_mean = bn_module.running_mean
        running_var = bn_module.running_var
        eps = bn_module.eps
        gamma = bn_module.weight  # BN 的可学习缩放参数 (weight)
        beta = bn_module.bias  # BN 的可学习偏移参数 (bias)

        if gamma is None:  # BN 可能未启用仿射变换 (affine=False)
            gamma = torch.ones_like(running_mean)
        if beta is None:
            beta = torch.zeros_like(running_mean)

        # 3. 计算融合后的权重和偏置
        # y = gamma * (x - mean) / sqrt(var + eps) + beta  (BN 公式)
        # y = gamma * w_conv * x_in + gamma * (b_conv - mean) / sqrt(var + eps) + beta (代入 Conv/Linear: x = w*x_in + b)
        # scale = gamma / sqrt(var + eps)
        # W_fused = w_conv * scale
        # b_fused = (b_conv - mean) * scale + beta

        scale = gamma / torch.sqrt(running_var + eps)

        # 调整 scale 的形状以匹配权重维度
        # Conv1D: (cout, cin/groups, kw) -> scale: (cout) -> reshape: (cout, 1, 1)
        # Conv2D: (cout, cin/groups, kh, kw) -> scale: (cout) -> reshape: (cout, 1, 1, 1)
        # Conv3D: (cout, cin/groups, kd, kh, kw) -> scale: (cout) -> reshape: (cout, 1, 1, 1, 1)
        # Linear: (cout, cin) -> scale: (cout) -> reshape: (cout, 1)
        scale_shape = [1] * w_conv.dim()
        scale_shape[0] = -1  # 第一维是输出通道，保持不变
        w_fused = w_conv * scale.reshape(scale_shape)

        if b_conv is not None:
            b_fused = (b_conv - running_mean) * scale + beta
        else:
            b_fused = (
                0.0 - running_mean
            ) * scale + beta  # 如果 Conv/Linear 没有偏置，则认为其偏置为 0

        # 4. 更新 Conv/Linear 模块的参数
        # 使用 Parameter 包装以确保它们仍然是可训练的参数
        conv_or_linear_module.weight = nn.Parameter(
            w_fused.detach()
        )  # detach() 以防万一
        # 确保创建新的 Parameter 对象，即使原来 bias 是 None
        conv_or_linear_module.bias = nn.Parameter(b_fused.detach())

        graph_utils_logger.debug(f"    权重和偏置已融合到 {conv_or_linear_name}")

        # 5. 修改 FX 图
        conv_node = nodes_by_target.get(conv_or_linear_name)
        bn_node = nodes_by_target.get(bn_name)

        if not conv_node or not bn_node:
            graph_utils_logger.error(
                f"    图修改错误：找不到节点 {conv_or_linear_name} 或 {bn_name}。"
            )
            continue

        try:
            # 将所有使用 BN 输出的地方，改为使用 Conv/Linear 的输出
            graph_utils_logger.debug(
                f"    将节点 '{_colorize(bn_node.name, COLOR_NODE)}' 的用户重定向到 '{_colorize(conv_node.name, COLOR_NODE)}'"
            )
            bn_node.replace_all_uses_with(conv_node)

            # 从图中删除 BN 节点
            graph_utils_logger.debug(
                f"    从图中删除节点 '{_colorize(bn_node.name, COLOR_NODE)}' (target={_colorize(str(bn_node.target), COLOR_MODULE)})"
            )
            model.graph.erase_node(bn_node)

        except Exception as e:
            graph_utils_logger.exception(
                f"    在修改图以融合 {conv_or_linear_name} 和 {bn_name} 时出错: {e}"
            )
            continue

        # 6. 从模型中删除 BatchNorm 模块
        try:
            _delete_module(model, bn_name)
            graph_utils_logger.debug(
                f"    从模型实例中删除模块 '{_colorize(bn_name, COLOR_MODULE)}'"
            )
        except Exception as e:
            graph_utils_logger.exception(f"    删除模块 {bn_name} 时出错: {e}")

        fused_count += 1

    if fused_count > 0:
        graph_utils_logger.info(f"成功融合了 {fused_count} 对 Conv/Linear-BN。")
        # 7. 清理图并重新编译
        try:
            # 删除可能产生的死代码（例如，如果BN节点删除后某些东西不再需要）
            model.graph.eliminate_dead_code()
            # 验证图的结构是否仍然有效
            model.graph.lint()
            # 使图的更改生效
            model.recompile()
            graph_utils_logger.info("图已清理并重新编译。")
        except Exception as e:
            graph_utils_logger.exception(f"融合后清理或重新编译图时出错: {e}")
            raise RuntimeError(f"融合后图处理失败: {e}") from e
    else:
        graph_utils_logger.info("没有实际执行融合操作。")

    return model  # 返回修改后的模型
