import torch.nn as nn
from dataclasses import dataclass
from typing import (
    Callable,
    Optional,
    Dict,
    Type,
    Union,
)

from .observer import PlaceholderObserver
from .fake_quant import PlaceholderFakeQuant


# 类型提示别名
# 定义一个类型别名，表示“一个返回 nn.Module 实例的可调用对象”
# 这通常用于指向 Observer 或 FakeQuant 类的构造函数
ObserverOrFakeQuantCls = Callable[[], nn.Module]


# --- QConfig ---
@dataclass
class QConfig:
    """
    量化配置类，用于指定激活和权重的 Observer 或 FakeQuant 模块工厂。

    此类是一个数据类 (dataclass)，简化了构造函数和属性定义。

    属性:
        activation: 一个返回 nn.Module 实例的 callable（零参数函数或类名）。
                    它负责创建用于量化/观察“激活”的模块实例。
                    通常是 Observer 类 (用于 PTQ) 或 FakeQuant 类 (用于 QAT)。
                    可以为 None，表示不量化激活。
        weight:     一个返回 nn.Module 实例的 callable（零参数函数或类名）。
                    它负责创建用于量化/观察“权重”的模块实例。
                    通常是 Observer 类 (用于 PTQ) 或 FakeQuant 类 (用于 QAT)。
                    可以为 None，表示不量化权重。
    """

    activation: Optional[ObserverOrFakeQuantCls]
    weight: Optional[ObserverOrFakeQuantCls]


# --- 默认 QConfig 示例 ---
# 使用上面定义的占位符模块创建一些默认配置

# 适用于 PTQ (Post Training Quantization) 的默认配置，使用占位符 Observer
default_placeholder_ptq_qconfig = QConfig(
    activation=PlaceholderObserver,  # 使用 PlaceholderObserver 类作为工厂
    weight=PlaceholderObserver,  # 使用 PlaceholderObserver 类作为工厂
)

# 适用于 QAT (Quantization Aware Training) 的默认配置，使用占位符 FakeQuant
default_placeholder_qat_qconfig = QConfig(
    activation=PlaceholderFakeQuant,  # 使用 PlaceholderFakeQuant 类作为工厂
    weight=PlaceholderFakeQuant,  # 使用 PlaceholderFakeQuant 类作为工厂
)

# 提供一个通用的默认值，用户可以根据需要选择 PTQ 或 QAT 版本
# 这里我们默认选择 PTQ 的占位符配置
default_placeholder_qconfig = default_placeholder_ptq_qconfig


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
            与键关联的 QConfig，如果未直接设置则返回 None。
        """
        if isinstance(key, str):
            return self._object_name_qconfigs.get(key)
        # 需要检查 key 是否真的是一个类型，并且是 nn.Module 的子类
        elif isinstance(key, type) and issubclass(key, nn.Module):
            return self._module_type_qconfigs.get(key)
        else:
            raise TypeError(f"不支持的键类型: {type(key)}")

    def get_qconfig(
        self, module_type: Type[nn.Module], object_name: Optional[str] = None
    ) -> Optional[QConfig]:
        """
        根据优先级规则获取给定模块的有效 QConfig。

        Args:
            module_type: 模块的类类型。
            object_name: 模块的 FQN（可选）。

        Returns:
            应用于此模块的 QConfig (可能是 None)，或全局 QConfig。
            如果所有级别都没有找到配置，则返回 None。
        """
        # 1. 按对象名称查找
        if object_name is not None and object_name in self._object_name_qconfigs:
            return self._object_name_qconfigs[
                object_name
            ]  # 如果值为 None，表示明确禁用

        # 2. 按模块类型查找
        # 更鲁棒的方式是检查 MRO (方法解析顺序)
        qconfig_found = None
        # 这里我们简化为直接匹配，但实际可能需要遍历 mro
        # for m_type in self._module_type_qconfigs:
        #    if issubclass(module_type, m_type):
        #        # 需要处理找到多个匹配的情况，例如取最具体的
        #        qconfig_found = self._module_type_qconfigs[m_type]
        #        break # 简化，找到第一个就用
        if module_type in self._module_type_qconfigs:
            qconfig_found = self._module_type_qconfigs[module_type]

        if qconfig_found is not None:  # 如果找到了，无论是 QConfig 还是 None，都用它
            return qconfig_found
        # 如果类型查找返回 None (即没有为该类型设置特定规则)，继续检查全局

        # 3. 返回全局配置
        return self._global_qconfig


# --- 实用函数 ---
def get_default_ptq_qconfig() -> QConfig:
    """返回默认的 PTQ QConfig (使用占位符 Observer)。"""
    return default_placeholder_ptq_qconfig


def get_default_qat_qconfig() -> QConfig:
    """返回默认的 QAT QConfig (使用占位符 FakeQuant)。"""
    return default_placeholder_qat_qconfig
