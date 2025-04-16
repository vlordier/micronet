from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .observer import (
    MinMaxObserver,
    _calculate_qmin_qmax,
)  # 导入 Observer 和辅助函数

# 类型提示：一个 Observer 类或其工厂函数
ObserverCls = Callable[..., nn.Module]


class FakeQuantize(nn.Module):
    """
    核心伪量化模块。

    该模块总是在 `prepare` 阶段被插入。它的行为通过内部状态控制：
    1. **观察/校准模式**: 激活内部 Observer 收集统计信息。forward 只传递数据。
    2. **伪量化模式**: 使用当前的 scale 和 zero_point 对输入进行伪量化
       (quantize-dequantize)。forward 返回伪量化后的数据。

    Scale 和 Zero Point 可以:
    - 通过调用 `calculate_qparams` 从 Observer 的统计数据中计算得到。
    - 设置为可学习参数 (learnable) 用于 QAT。
    """

    # 使用 slots 优化内存
    __slots__ = [
        "observer",
        "scale",
        "zero_point",
        "dtype",
        "qscheme",
        "quant_min",
        "quant_max",
        "eps",
        "observer_enabled",
        "fake_quant_enabled",
        "is_qat_learning",
    ]

    # 使用 observer_cls 而不是 observer_instance 来允许 QConfig 控制 Observer 类型
    def __init__(
        self,
        observer_cls: ObserverCls = MinMaxObserver,
        dtype: torch.dtype = torch.quint8,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        reduce_range: bool = False,
        # 可以添加 observer 特定的参数
        observer_kwargs: Optional[dict] = None,
        # 默认 scale=1, zp=0
        initial_scale: float = 1.0,
        initial_zero_point: int = 0,
        eps: float = torch.finfo(torch.float32).eps,
    ):
        """
        初始化 FakeQuantize 模块。

        Args:
            observer_cls (ObserverCls): 用于统计信息收集的 Observer 类或工厂函数。
                                        默认为 MinMaxObserver。
            dtype (torch.dtype): 量化的目标数据类型。默认为 torch.quint8。
            qscheme (torch.qscheme): 量化方案。默认为 torch.per_tensor_affine。
            reduce_range (bool): 是否减少量化范围。默认为 False。
            observer_kwargs (Optional[dict]): 传递给 Observer 构造函数的额外参数。
            initial_scale (float): Scale 参数的初始值。默认为 1.0。
            initial_zero_point (int): Zero Point 参数的初始值。默认为 0。
            eps (float): 用于伪量化计算中避免除零的小值。
        """
        super().__init__()

        if observer_kwargs is None:
            observer_kwargs = {}

        # 实例化内部 Observer
        self.observer = observer_cls(
            dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, **observer_kwargs
        )

        self.dtype = dtype
        self.qscheme = qscheme
        self.eps = eps

        # 获取量化范围
        self.quant_min, self.quant_max = _calculate_qmin_qmax(self.dtype)

        # 初始化 scale 和 zero_point 为可学习参数 (nn.Parameter)
        # 即使在 PTQ 中，也将它们设为 Parameter，只是 requires_grad=False
        self.scale = Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        # Zero point 在 QAT 中通常也是浮点学习，但在 PTQ 后是整数。
        # 这里统一用浮点 Parameter 存储，在需要整数时进行转换。
        self.zero_point = Parameter(
            torch.tensor(initial_zero_point, dtype=torch.float32)
        )

        # 内部状态标志
        self.observer_enabled: bool = True  # 默认开启观察模式 (用于 PTQ 校准)
        self.fake_quant_enabled: bool = False  # 默认关闭伪量化
        self.is_qat_learning: bool = False  # 默认关闭参数学习

        # 根据初始状态设置参数的可学习性
        self._update_param_learning_state()

    def _update_param_learning_state(self):
        """根据 is_qat_learning 状态更新 scale 和 zero_point 的 requires_grad"""
        self.scale.requires_grad_(self.is_qat_learning)
        self.zero_point.requires_grad_(self.is_qat_learning)

    def enable_observer(self, enabled: bool = True) -> "FakeQuantize":
        """启用或禁用内部 Observer 的统计收集。"""
        self.observer_enabled = enabled
        # print(f"  [FakeQuant {id(self)}] Observer {'启用' if enabled else '禁用'}")
        return self

    def enable_fake_quant(self, enabled: bool = True) -> "FakeQuantize":
        """启用或禁用伪量化操作。"""
        self.fake_quant_enabled = enabled
        # print(f"  [FakeQuant {id(self)}] FakeQuant {'启用' if enabled else '禁用'}")
        return self

    def enable_learning(
        self, enabled: bool = True, inherit_qparams: bool = True
    ) -> "FakeQuantize":
        """
        启用或禁用 scale/zero_point 的学习（QAT 模式）。

        Args:
            enabled (bool): 是否启用学习。
            inherit_qparams (bool): 如果启用学习，应该首先尝试从 Observer
                                    继承当前的 qparams 作为初始值。如果为 False，
                                    或者 Observer 还没有有效的 qparams，
                                    将使用当前的 scale/zero_point 值。
                                    默认为 True。

        Returns:
            FakeQuantize: self
        """
        self.is_qat_learning = enabled
        # print(f"  [FakeQuant {id(self)}] 参数学习 {'启用' if enabled else '禁用'}")

        if enabled and inherit_qparams:
            # 尝试从 observer 获取 qparams 并更新当前参数
            try:
                # 注意：这里只是获取 observer 计算出的值，不改变 observer 状态
                scale, zp = self.observer.calculate_qparams()
                # 只有在 observer 有效统计数据时才更新 (min != inf, max != -inf)
                if torch.isfinite(self.observer.min_val) and torch.isfinite(
                    self.observer.max_val
                ):
                    with torch.no_grad():
                        self.scale.copy_(scale)
                        # 将整数 zp 转换为浮点赋给 Parameter
                        self.zero_point.copy_(zp.to(torch.float32))
                    # print(f"  [FakeQuant {id(self)}] QAT 学习已启用，并从 Observer 继承 qparams: scale={scale.item():.4f}, zp={zp.item()}")
                # else:
                # print(f"  [FakeQuant {id(self)}] QAT 学习已启用，但 Observer 统计无效，使用现有参数: scale={self.scale.item():.4f}, zp={self.zero_point.item():.1f}")

            except Exception as e:
                # print(f"  [FakeQuant {id(self)}] QAT 学习已启用，但从 Observer 继承 qparams 失败 ({e})，使用现有参数: scale={self.scale.item():.4f}, zp={self.zero_point.item():.1f}")
                pass  # 失败则保持现有参数

        self._update_param_learning_state()  # 更新参数的 requires_grad 状态
        return self

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        让内部 Observer 计算 qparams，并用计算结果更新本模块的 scale 和 zero_point。
        通常在 PTQ 校准后调用。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 更新后的 scale 和 zero_point。
        """
        scale, zp = self.observer.calculate_qparams()
        with torch.no_grad():  # 更新参数值，但不影响梯度记录
            self.scale.copy_(scale)
            self.zero_point.copy_(zp.to(torch.float32))  # 存储为浮点
        # print(f"  [FakeQuant {id(self)}] 已从 Observer 更新 qparams: scale={self.scale.item():.4f}, zp={self.zero_point.item():.1f}")
        return self.scale, self.zero_point

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        模块的前向传播逻辑。

        根据内部状态执行操作：
        - 如果 observer_enabled=True: 调用 observer.forward(X) 并返回 X。
        - 如果 fake_quant_enabled=True: 执行伪量化并返回结果。
        - 否则: 返回 X (不推荐的状态)。
        """
        if self.observer_enabled:
            # 观察模式：收集统计数据，不改变数据
            self.observer(X)
            return X
        elif self.fake_quant_enabled:
            # 伪量化模式：应用 quantize-dequantize
            # --- 核心伪量化逻辑 ---
            # 1. 将 zero_point (可能是浮点) 转换为整数，用于计算
            #    这里根据 qscheme 决定如何处理 zero_point
            if self.qscheme == torch.per_tensor_symmetric:
                # 对称量化，强制内部计算使用 0 作为零点
                zero_point_int = torch.tensor(0, dtype=torch.int64, device=X.device)
            else:  # 非对称 torch.per_tensor_affine
                # 非对称量化，使用 self.zero_point，四舍五入到最近整数并 clamp
                zero_point_int = torch.round(self.zero_point.data).to(torch.int64)
                zero_point_int = torch.clamp(
                    zero_point_int, self.quant_min, self.quant_max
                )

            # 2. 量化: (X / scale) + zero_point
            #    使用 Tensor Lateny 库中类似实现：torch.clamp(torch.round(X / self.scale) + zero_point_int, ...)
            X_q = torch.round(X / (self.scale + self.eps) + zero_point_int)

            # 3. Clamp 到量化范围 [qmin, qmax]
            X_q_clamped = torch.clamp(X_q, self.quant_min, self.quant_max)

            # 4. 反量化: (X_q_clamped - zero_point) * scale
            #    使用原始的（可能是浮点的）zero_point 进行反量化，以匹配 QAT 行为
            X_dq = (X_q_clamped - self.zero_point) * self.scale

            # print(f"  [FakeQuant {id(self)}] Applied FakeQuant: scale={self.scale.item():.4f}, zp={self.zero_point.item():.1f}")
            return X_dq
        else:
            # 如果两者都禁用，则直接返回输入 (可能用于调试或特定情况)
            # print(f"  [FakeQuant {id(self)}] Warning: Observer 和 FakeQuant 都被禁用，直接传递数据。")
            return X

    def extra_repr(self) -> str:
        """提供更详细的模块表示。"""
        obs_repr = repr(self.observer)
        state = []
        if self.observer_enabled:
            state.append("OBSERVING")
        if self.fake_quant_enabled:
            state.append("FAKE_QUANT")
        if self.is_qat_learning:
            state.append("LEARNING")
        if not state:
            state.append("INACTIVE")

        return f"state=[{', '.join(state)}], scale={self.scale.item():.4f}, zero_point={self.zero_point.item():.4f}, observer=({obs_repr})"

    def reset_observer_stats(self):
        """方便地重置内部 observer 的统计信息。"""
        if hasattr(self.observer, "reset_stats"):
            self.observer.reset_stats()
