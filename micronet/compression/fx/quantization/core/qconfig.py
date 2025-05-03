# micronet/compression/fx/quantization/core/qconfig.py

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

    这允许对模型的不同部分应用不同的量化策略。

    映射查找优先级（从高到低）:
    1. 对象名称 (object_name_qconfigs): 完全匹配模块的 FQN (Fully Qualified Name)。
    2. 模块类型 (module_type_qconfigs): 匹配模块的类类型。
    3. 全局设置 (global_qconfig): 如果以上均未匹配，则使用全局配置。
    """

    def __init__(self):
        """初始化空的 QConfigMapping。"""
        self._global_qconfig: Optional[QConfig] = None
        self._module_type_qconfigs: Dict[Type[nn.Module], Optional[QConfig]] = {}
        self._object_name_qconfigs: Dict[str, Optional[QConfig]] = {}  # FQN -> QConfig

    def set_global(self, qconfig: Optional[QConfig]) -> "QConfigMapping":
        """
        设置全局 QConfig。

        Args:
            qconfig: 要设置的全局 QConfig，或 None 以移除全局配置。

        Returns:
            self，允许链式调用。
        """
        self._global_qconfig = qconfig
        return self

    def set_module_type(
        self, module_type: Type[nn.Module], qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        """
        为特定的模块类型设置 QConfig。

        Args:
            module_type: 要配置的模块类 (例如 nn.Linear, nn.Conv2d)。
            qconfig: 要应用于此模块类型的 QConfig，或 None 以移除此类型的特定配置。

        Returns:
            self，允许链式调用。
        """
        self._module_type_qconfigs[module_type] = qconfig
        return self

    def set_object_name(
        self, object_name: str, qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        """
        为特定名称的模块（使用其完全限定名 FQN）设置 QConfig。

        Args:
            object_name: 模块的 FQN 字符串 (例如 'encoder.layer.0.attention')。
            qconfig: 要应用于此名称模块的 QConfig，或 None 以禁用此模块的量化。

        Returns:
            self，允许链式调用。
        """
        self._object_name_qconfigs[object_name] = qconfig
        return self

    def __getitem__(self, key: Union[Type[nn.Module], str]) -> Optional[QConfig]:
        """
        允许使用方括号语法获取特定模块类型或对象名称的 QConfig。
        注意：此方法不执行完整的优先级查找，仅用于直接访问已设置的配置。
              要获取给定模块的有效 QConfig，请使用 `get_qconfig` 方法。

        Args:
            key: 模块类型或对象名称（FQN 字符串）。

        Returns:
            与键关联的 QConfig，如果未直接设置则抛出 KeyError。
        """
        if isinstance(key, str):
            if key in self._object_name_qconfigs:
                return self._object_name_qconfigs[key]
            else:
                raise KeyError(f"对象名称 '{key}' 未在映射中直接设置。")
        elif isinstance(key, type) and issubclass(key, nn.Module):
            if key in self._module_type_qconfigs:
                return self._module_type_qconfigs[key]
            else:
                raise KeyError(f"模块类型 '{key.__name__}' 未在映射中直接设置。")
        else:
            raise TypeError(f"不支持的键类型: {type(key)}")

    def get_qconfig(
        self, module_type: Type[nn.Module], object_name: Optional[str] = None
    ) -> Optional[QConfig]:
        """
        根据优先级规则获取给定模块的有效 QConfig。

        优先级顺序 (从高到低):
        1. 按对象名称 (FQN) 查找精确匹配。
        2. 按模块类型查找。
        3. 使用全局 QConfig。

        如果一个模块在某个优先级级别被显式配置为 None，则表示禁用该模块的量化，
        查找过程会停止并返回 None，不会继续查找更低优先级的配置。

        Args:
            module_type: 模块的类类型。
            object_name: 模块的 FQN（可选）。

        Returns:
            应用于此模块的 QConfig (可能是 None)。如果所有级别都没有找到匹配，
            或者找到了明确的 None 配置，则返回 None。
        """
        # 优先级 1: 按对象名称查找
        if object_name is not None and object_name in self._object_name_qconfigs:
            # 如果找到，立即返回 (可能是 QConfig 实例或 None)
            return self._object_name_qconfigs[object_name]

        # 优先级 2: 按模块类型查找
        # (这里简化为直接匹配，更复杂的实现可以考虑 MRO)
        if module_type in self._module_type_qconfigs:
            # 如果找到，立即返回 (可能是 QConfig 实例或 None)
            return self._module_type_qconfigs[module_type]

        # 如果以上级别都没有找到明确的配置 (包括 None)

        # 优先级 3: 返回全局配置
        return self._global_qconfig


# --- 实用函数 ---
def get_default_ptq_qconfig() -> QConfig:
    """返回默认的 PTQ QConfig (使用 FakeQuantize + MinMaxObserver)。"""
    return default_ptq_qconfig


def get_default_qat_qconfig() -> QConfig:
    """返回默认的 QAT QConfig (使用 FakeQuantize + MinMaxObserver, 配置用于 QAT)。"""
    return default_qat_qconfig
