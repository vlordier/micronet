import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import (
    Callable,
    Optional,
    Dict,
    Type,
    Union,
)
import functools

from .observer import MinMaxObserver
from .fake_quant import FakeQuantize

# 类型提示别名 - 指向 FakeQuantize 类或其工厂
QuantizerClsFactory = Callable[[], nn.Module]


# --- QConfig ---
@dataclass
class QConfig:
    """
    量化配置类，用于指定激活和权重的 FakeQuantize 模块工厂。

    activation 和 weight 应该总是指向一个返回 FakeQuantize 实例的工厂。
    工厂函数可以通过 functools.partial 或 lambda 来预设 FakeQuantize 的参数
    （例如，使用的 Observer 类型、dtype、qscheme 等）。

    属性:
        activation: 返回 FakeQuantize 实例的 callable。None 表示不量化激活。
        weight:     返回 FakeQuantize 实例的 callable。None 表示不量化权重。
    """

    activation: Optional[QuantizerClsFactory]
    weight: Optional[QuantizerClsFactory]


# --- 默认 QConfig 示例 ---

# 适用于 PTQ 的默认配置：
# 使用 FakeQuantize，内部包含 MinMaxObserver，dtype 为 quint8 (激活) / qint8 (权重)
# 注意：权重通常推荐对称量化
default_ptq_qconfig = QConfig(
    activation=functools.partial(
        FakeQuantize,
        observer_cls=MinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
    weight=functools.partial(
        FakeQuantize,
        observer_cls=MinMaxObserver,
        dtype=torch.qint8,  # 权重通常用 qint8
        qscheme=torch.per_tensor_symmetric,  # 对称量化
        reduce_range=False,
    ),
)

# 适用于 QAT 的默认配置：
# 结构上与 PTQ 相同（都使用 FakeQuantize），但在训练循环中会配置 FakeQuantize
# 进入学习模式 (enable_learning=True)。
# 这里可以定义不同的初始设置，例如不同的 observer 或 qscheme，但通常从 PTQ 的配置开始。
default_qat_qconfig = QConfig(
    activation=functools.partial(
        FakeQuantize,
        observer_cls=MinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
    weight=functools.partial(
        FakeQuantize,
        observer_cls=MinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    ),
)

# 提供一个通用的默认值，默认为 PTQ 配置
default_qconfig = default_ptq_qconfig


# --- QConfigMapping ---
class QConfigMapping:
    """
    管理不同模块类型或模块实例名称与 QConfig 之间映射的容器。
    (此类结构和逻辑基本保持不变)
    """

    def __init__(self):
        self._global_qconfig: Optional[QConfig] = None
        self._module_type_qconfigs: Dict[Type[nn.Module], Optional[QConfig]] = {}
        self._object_name_qconfigs: Dict[str, Optional[QConfig]] = {}

    def set_global(self, qconfig: Optional[QConfig]) -> "QConfigMapping":
        self._global_qconfig = qconfig
        return self

    def set_module_type(
        self, module_type: Type[nn.Module], qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        self._module_type_qconfigs[module_type] = qconfig
        return self

    def set_object_name(
        self, object_name: str, qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        self._object_name_qconfigs[object_name] = qconfig
        return self

    def __getitem__(self, key: Union[Type[nn.Module], str]) -> Optional[QConfig]:
        if isinstance(key, str):
            return self._object_name_qconfigs.get(key)
        elif isinstance(key, type) and issubclass(key, nn.Module):
            # 简单查找，可以优化为检查父类
            return self._module_type_qconfigs.get(key)
        else:
            raise TypeError(f"不支持的键类型: {type(key)}")

    def get_qconfig(
        self, module_type: Type[nn.Module], object_name: Optional[str] = None
    ) -> Optional[QConfig]:
        # 优先级: object_name > module_type > global
        if object_name is not None and object_name in self._object_name_qconfigs:
            return self._object_name_qconfigs[object_name]

        # 简单类型匹配 (可以改进为检查 MRO)
        qconfig_found = self._module_type_qconfigs.get(module_type)
        if qconfig_found is not None:
            return qconfig_found

        return self._global_qconfig


# --- 实用函数 ---
def get_default_ptq_qconfig() -> QConfig:
    """返回默认的 PTQ QConfig (使用 FakeQuantize + MinMaxObserver)。"""
    return default_ptq_qconfig


def get_default_qat_qconfig() -> QConfig:
    """返回默认的 QAT QConfig (使用 FakeQuantize + MinMaxObserver, 配置用于 QAT)。"""
    return default_qat_qconfig
