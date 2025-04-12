# quantization_framework/quant_core/qconfig.py

import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Type, Union, Any

# --- Placeholder Modules ---
# 这些模块在 prepare 阶段被插入，作为未来 Observer 或 FakeQuant 的占位符
# 它们本身在 prepare 阶段不做任何计算，只是标记位置


class PlaceholderObserver(nn.Module):
    """
    一个简单的占位符模块，标记将来要插入 Observer 的位置。
    在 PTQ 的 prepare 阶段使用。
    """

    def __init__(self):
        super().__init__()
        # 在实际的 Observer 中，这里会初始化统计量 buffer
        pass

    def forward(self, x):
        # 实际 Observer 会在这里收集统计数据
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class PlaceholderFakeQuant(nn.Module):
    """
    一个简单的占位符模块，标记将来要插入 FakeQuantize 模块的位置。
    在 QAT 的 prepare 阶段使用。
    """

    def __init__(self):
        super().__init__()
        # 在实际的 FakeQuant 中，这里会初始化 scale/zero_point 参数
        # 并且这些参数可能是可学习的
        pass

    def forward(self, x):
        # 实际 FakeQuant 会在这里执行模拟量化操作
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# 类型提示别名
ObserverOrFakeQuantCls = Callable[[], nn.Module]


# --- QConfig ---
@dataclass
class QConfig:
    """
    量化配置类，指定用于激活和权重的 Observer 或 FakeQuant 模块。

    属性:
        activation: 一个返回 nn.Module 实例的 callable (通常是 Observer 或 FakeQuant 类)。
                    用于量化激活值。
        weight: 一个返回 nn.Module 实例的 callable (通常是 Observer 或 FakeQuant 类)。
                用于量化权重。
    """

    activation: Optional[ObserverOrFakeQuantCls]
    weight: Optional[ObserverOrFakeQuantCls]


# --- QConfigMapping ---
class QConfigMapping:
    """
    管理不同模块类型或模块实例名称与 QConfig 之间映射的容器。

    映射优先级:
    1. 对象名称 (object_name_qconfigs)
    2. 模块类型 (module_type_qconfigs)
    3. 全局设置 (global_qconfig)
    """

    def __init__(self):
        self._global_qconfig: Optional[QConfig] = None
        self._module_type_qconfigs: Dict[Type[nn.Module], QConfig] = {}
        self._object_name_qconfigs: Dict[str, QConfig] = (
            {}
        )  # 按模块的 FQN (Fully Qualified Name) 映射

    def set_global(self, qconfig: Optional[QConfig]) -> "QConfigMapping":
        """设置全局 QConfig。"""
        self._global_qconfig = qconfig
        return self

    def set_module_type(
        self, module_type: Type[nn.Module], qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        """为特定的模块类型设置 QConfig。"""
        if qconfig is None:
            self._module_type_qconfigs.pop(module_type, None)
        else:
            self._module_type_qconfigs[module_type] = qconfig
        return self

    def set_object_name(
        self, object_name: str, qconfig: Optional[QConfig]
    ) -> "QConfigMapping":
        """为特定名称的模块（FQN）设置 QConfig。"""
        if qconfig is None:
            self._object_name_qconfigs.pop(object_name, None)
        else:
            self._object_name_qconfigs
