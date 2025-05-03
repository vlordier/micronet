# micronet/compression/fx/quantization/tests/core/test_graph_utils.py

import pytest
import torch
import torch.nn as nn
import copy
from typing import Callable, Type

from micronet.compression.fx.quantization.core.graph_utils import (
    is_quantizable_weight_module,
    is_quantizable_activation_module,
    is_quantizable_activation_function,
    is_quantizable_activation_method,
    _get_nested_module,
    _delete_module,
    DEFAULT_WEIGHT_QUANT_MODULE_TYPES,
    DEFAULT_ACTIVATION_QUANT_MODULE_TYPES,
    DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES,
    DEFAULT_ACTIVATION_QUANT_METHOD_NAMES,
)


# --- 测试 is_quantizable_weight_module ---


@pytest.mark.parametrize("module_cls", DEFAULT_WEIGHT_QUANT_MODULE_TYPES)
def test_is_quantizable_weight_module_positive(module_cls: Type[nn.Module]):
    """测试默认列表中预期需要量化权重的模块"""
    # 需要为模块提供合适的初始化参数
    if module_cls in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        module = module_cls(3, 6, 3)
    elif module_cls == nn.Linear:
        module = module_cls(10, 5)
    elif module_cls == nn.Embedding:
        module = module_cls(10, 3)
    elif module_cls == nn.EmbeddingBag:
        module = module_cls(10, 3, mode="sum")  # EmbeddingBag 需要 mode
    elif module_cls in [nn.LSTM, nn.GRU]:
        module = module_cls(10, 20, 2)
    elif module_cls in [nn.LSTMCell, nn.GRUCell]:
        module = module_cls(10, 20)
    elif module_cls in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
        module = module_cls(6)  # num_features
    else:
        # 如果添加了新的默认类型，需要在这里添加初始化逻辑
        pytest.skip(f"测试用例未针对 {module_cls} 提供初始化参数")

    assert (
        is_quantizable_weight_module(module) is True
    ), f"{module_cls} 应该被识别为权重可量化"


# Negative test 测试不在列表中的情况
@pytest.mark.parametrize(
    "module_cls",
    [
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout,
        nn.Sequential,
        nn.Module,  # 基类
    ],
)
def test_is_quantizable_weight_module_negative(module_cls: Type[nn.Module]):
    """测试不应被识别为权重可量化的模块"""
    if module_cls == nn.MaxPool2d:
        module = module_cls(kernel_size=2)
    elif module_cls == nn.Dropout:
        module = module_cls(p=0.5)
    elif module_cls == nn.Sequential:
        module = module_cls(nn.ReLU())
    elif module_cls == nn.ReLU:
        module = module_cls()
    elif module_cls == nn.Module:
        module = module_cls()
    else:
        pytest.skip(f"测试用例未针对 {module_cls} 提供初始化参数")

    assert (
        is_quantizable_weight_module(module) is False
    ), f"{module_cls} 不应被识别为权重可量化"


# --- 测试 is_quantizable_activation_module ---


@pytest.mark.parametrize("module_cls", DEFAULT_ACTIVATION_QUANT_MODULE_TYPES)
def test_is_quantizable_activation_module_positive(module_cls: Type[nn.Module]):
    """测试默认列表中预期需要量化其输出激活的模块"""
    if module_cls in [
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    ]:
        module = module_cls(3, 6, 3)
    elif module_cls == nn.Linear:
        module = module_cls(10, 5)
    elif module_cls == nn.Embedding:
        module = module_cls(10, 3)
    elif module_cls == nn.EmbeddingBag:
        module = module_cls(10, 3, mode="sum")
    elif module_cls in [
        nn.ReLU,
        nn.ReLU6,
        nn.Sigmoid,
        nn.Tanh,
        nn.Hardswish,
        nn.SiLU,
        nn.GELU,
        nn.ELU,
        nn.LeakyReLU,
        nn.PReLU,
        nn.Softplus,
        nn.Softsign,
        nn.Tanhshrink,
        nn.Softmin,
        nn.Softmax,
        nn.LogSoftmax,
        nn.Identity,
    ]:
        # PReLU 需要 num_parameters
        if module_cls == nn.PReLU:
            module = module_cls(num_parameters=6)  # 假设通道数为 6
        elif module_cls in [nn.Softmax, nn.LogSoftmax, nn.Softmin]:
            module = module_cls(dim=1)  # 需要 dim 参数
        else:
            module = module_cls()
    elif module_cls in [
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.MaxPool3d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AvgPool3d,
        nn.FractionalMaxPool2d,
        nn.LPPool1d,
        nn.LPPool2d,
    ]:
        if module_cls == nn.FractionalMaxPool2d:
            module = module_cls(kernel_size=2, output_ratio=0.5)
        elif module_cls in [nn.LPPool1d, nn.LPPool2d]:
            module = module_cls(norm_type=2, kernel_size=2)
        else:
            module = module_cls(kernel_size=2)
    elif module_cls in [
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d,
        nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    ]:
        module = module_cls(output_size=1)
    elif module_cls in [
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.SyncBatchNorm,
        nn.LocalResponseNorm,
    ]:
        # BN/LN/IN/GN/SyncBN 需要维度信息
        if module_cls in [nn.BatchNorm1d, nn.InstanceNorm1d]:
            module = module_cls(6)  # num_features
        elif module_cls in [nn.BatchNorm2d, nn.InstanceNorm2d]:
            module = module_cls(6)  # num_features
        elif module_cls in [nn.BatchNorm3d, nn.InstanceNorm3d]:
            module = module_cls(6)  # num_features
        elif module_cls == nn.LayerNorm:
            module = module_cls(normalized_shape=[10])  # 示例形状
        elif module_cls == nn.GroupNorm:
            module = module_cls(
                num_groups=3, num_channels=6
            )  # 需要 num_groups 和 num_channels
        elif module_cls == nn.SyncBatchNorm:
            # SyncBatchNorm 比较特殊，可能需要在分布式环境中测试，这里简化
            module = module_cls(6)
        elif module_cls == nn.LocalResponseNorm:
            module = module_cls(size=5)  # 需要 size
        else:
            pytest.skip(f"未处理的 Norm 类型初始化: {module_cls}")
    elif module_cls in [
        nn.Upsample,
        nn.UpsamplingNearest2d,
        nn.UpsamplingBilinear2d,
    ]:
        # Upsample 需要 size 或 scale_factor
        module = module_cls(scale_factor=2)
    elif module_cls in [nn.PixelShuffle, nn.PixelUnshuffle]:  # 添加 Pixel Shuffle
        module = module_cls(upscale_factor=2)  # PixelShuffle 需要 upscale_factor
        # PixelUnshuffle 需要 downscale_factor
        if module_cls == nn.PixelUnshuffle:
            module = module_cls(downscale_factor=2)
    elif module_cls == nn.Flatten:
        module = module_cls()  # 通常不需要参数
    elif module_cls == nn.Unflatten:
        module = module_cls(
            dim=1, unflattened_size=(2, 3)
        )  # 需要 dim 和 unflattened_size
    elif module_cls in [
        nn.LSTM,
        nn.GRU,
        nn.LSTMCell,
        nn.GRUCell,
    ]:
        if module_cls in [nn.LSTM, nn.GRU]:
            module = module_cls(10, 20, 2)
        else:  # Cells
            module = module_cls(10, 20)
    else:
        pytest.skip(f"测试用例未针对 {module_cls} 提供初始化参数")

    assert (
        is_quantizable_activation_module(module) is True
    ), f"{module_cls} 应该被识别为激活可量化"


# Negative test
@pytest.mark.parametrize(
    "module_cls",
    [
        nn.Dropout,
        nn.Sequential,
        nn.ModuleList,
        nn.ModuleDict,
        nn.Module,  # 基类 (虽然技术上 Identity 在上面，但这里保留基类作为反例)
    ],
)
def test_is_quantizable_activation_module_negative(module_cls: Type[nn.Module]):
    """测试不应被识别为激活可量化的模块"""
    if module_cls == nn.Dropout:
        module = module_cls(p=0.5)
    elif module_cls == nn.Sequential:
        module = module_cls(nn.ReLU())
    elif module_cls == nn.ModuleList:
        module = module_cls([nn.ReLU()])
    elif module_cls == nn.ModuleDict:
        module = module_cls({"act": nn.ReLU()})
    elif module_cls == nn.Module:
        module = module_cls()
    else:
        pytest.skip(f"未处理的模块类型初始化: {module_cls}")
    assert (
        is_quantizable_activation_module(module) is False
    ), f"{module_cls} 不应被识别为激活可量化"


# --- 测试 is_quantizable_activation_function ---


@pytest.mark.parametrize("func", DEFAULT_ACTIVATION_QUANT_FUNCTION_TYPES)
def test_is_quantizable_activation_function_positive(func: Callable):
    """测试默认列表中预期需要量化其输出的函数/操作符"""
    assert (
        is_quantizable_activation_function(func) is True
    ), f"{func} 应该被识别为激活可量化函数"


# Negative test
@pytest.mark.parametrize(
    "func",
    [
        torch.randn,
        print,
        lambda x: x + 1,
        nn.Conv2d,
        torch.Tensor.view,
    ],
)
def test_is_quantizable_activation_function_negative(func: Callable):
    """测试不应被识别为激活可量化函数"""
    assert (
        is_quantizable_activation_function(func) is False
    ), f"{func} 不应被识别为激活可量化函数"


# --- 测试 is_quantizable_activation_method ---


@pytest.mark.parametrize("method_name", DEFAULT_ACTIVATION_QUANT_METHOD_NAMES)
def test_is_quantizable_activation_method_positive(method_name: str):
    """测试默认列表中预期需要量化其输出的方法名"""
    assert (
        is_quantizable_activation_method(method_name) is True
    ), f"方法 '{method_name}' 应该被识别为激活可量化"


# Negative test
@pytest.mark.parametrize(
    "method_name",
    [
        "forward",
        "to",
        "state_dict",
        "parameters",
        "__init__",
        "my_custom_method",
    ],
)
def test_is_quantizable_activation_method_negative(method_name: str):
    """测试不应被识别为激活可量化的方法名"""
    assert (
        is_quantizable_activation_method(method_name) is False
    ), f"方法 '{method_name}' 不应被识别为激活可量化"


# --- 测试 _get_nested_module ---


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 6, 3), nn.ReLU())
        self.fc = nn.Linear(10, 5)
        self.empty_seq = nn.Sequential()


@pytest.fixture
def nested_model_instance():
    """提供一个嵌套模型的实例"""
    return NestedModel()


def test_get_nested_module_top_level(nested_model_instance):
    module = _get_nested_module(nested_model_instance, "layer1")
    assert isinstance(module, nn.Sequential)
    assert module is nested_model_instance.layer1

    module_fc = _get_nested_module(nested_model_instance, "fc")
    assert isinstance(module_fc, nn.Linear)
    assert module_fc is nested_model_instance.fc


def test_get_nested_module_nested(nested_model_instance):
    module_conv = _get_nested_module(nested_model_instance, "layer1.0")
    assert isinstance(module_conv, nn.Conv2d)
    assert module_conv is nested_model_instance.layer1[0]

    module_relu = _get_nested_module(nested_model_instance, "layer1.1")
    assert isinstance(module_relu, nn.ReLU)
    assert module_relu is nested_model_instance.layer1[1]


def test_get_nested_module_non_existent(nested_model_instance):
    module = _get_nested_module(nested_model_instance, "layer2")
    assert module is None

    module_nested = _get_nested_module(nested_model_instance, "layer1.2")
    assert module_nested is None

    module_invalid = _get_nested_module(nested_model_instance, "layer1.conv")
    assert module_invalid is None


def test_get_nested_module_empty_target(nested_model_instance):
    assert _get_nested_module(nested_model_instance, "") is None


def test_get_nested_module_on_empty_sequential(nested_model_instance):
    module = _get_nested_module(nested_model_instance, "empty_seq.0")
    assert module is None


# --- 测试 _delete_module ---


def test_delete_module_top_level(nested_model_instance):
    model_copy = copy.deepcopy(nested_model_instance)
    assert hasattr(model_copy, "layer1")
    _delete_module(model_copy, "layer1")
    assert not hasattr(model_copy, "layer1"), "模块 'layer1' 未被删除"
    assert hasattr(model_copy, "fc")


def test_delete_module_nested(nested_model_instance):
    model_copy = copy.deepcopy(nested_model_instance)
    assert hasattr(model_copy.layer1, "0")
    assert isinstance(model_copy.layer1[0], nn.Conv2d)

    _delete_module(model_copy, "layer1.0")

    remaining_children_names = [name for name, _ in model_copy.layer1.named_children()]
    assert "0" not in remaining_children_names, "子模块 'layer1.0' 未被删除"

    assert hasattr(model_copy, "layer1")
    assert "1" in remaining_children_names


def test_delete_module_non_existent_top(nested_model_instance):
    model_copy = copy.deepcopy(nested_model_instance)
    initial_modules = set(n for n, _ in model_copy.named_children())
    _delete_module(model_copy, "layer3")
    final_modules = set(n for n, _ in model_copy.named_children())
    assert initial_modules == final_modules


def test_delete_module_non_existent_nested(nested_model_instance):
    model_copy = copy.deepcopy(nested_model_instance)
    initial_layer1_children = set(n for n, _ in model_copy.layer1.named_children())
    _delete_module(model_copy, "layer1.2")
    final_layer1_children = set(n for n, _ in model_copy.layer1.named_children())
    assert initial_layer1_children == final_layer1_children


def test_delete_module_invalid_parent(nested_model_instance):
    model_copy = copy.deepcopy(nested_model_instance)
    initial_modules = set(n for n, _ in model_copy.named_children())
    _delete_module(model_copy, "non_existent_parent.child")
    final_modules = set(n for n, _ in model_copy.named_children())
    assert initial_modules == final_modules
