# micronet/compression/fx/quantization/core/observer.py

from typing import Tuple

import torch
import torch.nn as nn

from .quant_utils import calculate_qmin_qmax


class MinMaxObserver(nn.Module):
    """
    一个实际的 MinMax Observer 模块，用于 PTQ 校准阶段或 QAT 的统计初始化。

    它通过 forward 传递来观察输入张量的最小值和最大值，
    并存储这些统计数据。它不改变流经它的数据。
    `calculate_qparams` 方法用于在校准后根据收集到的 min/max
    值计算量化参数（scale 和 zero_point）。
    """

    # 使用 slots 优化内存
    __slots__ = ["min_val", "max_val", "dtype", "qscheme", "reduce_range", "eps"]

    def __init__(
        self,
        dtype: torch.dtype = torch.quint8,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        reduce_range: bool = False,
        eps: float = torch.finfo(torch.float32).eps,
    ):
        """
        初始化 MinMax Observer。

        Args:
            dtype (torch.dtype, optional): 目标量化数据类型。
                                           决定了量化范围 (qmin, qmax)。
                                           默认为 torch.quint8 (非对称)。
            qscheme (torch.qscheme, optional): 量化方案。
                                               通常是 torch.per_tensor_affine (非对称)
                                               或 torch.per_tensor_symmetric (对称)。
                                               默认为 torch.per_tensor_affine。
            reduce_range (bool, optional): 是否减少量化范围以避免零点误差。
                                           通常在对称量化中使用。默认为 False。
            eps (float, optional): 用于避免除以零的小值。
                                    默认为 float32 的最小值。
        """
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.reduce_range = reduce_range
        self.eps = eps

        # 注册 buffer 用于存储统计值，它们会被 state_dict 保存但不是模型参数
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:
        """
        观察输入张量并更新 min/max 统计值。

        Args:
            x_orig (torch.Tensor): 输入的浮点张量。

        Returns:
            torch.Tensor: 未经修改的原始输入张量。
        """
        x = x_orig.detach()  # 通常 observer 和模型在同一设备

        if x.numel() == 0:  # 处理空张量
            return x_orig

        # 计算当前批次的 min/max
        # per-tensor
        current_min = torch.min(x)
        current_max = torch.max(x)

        # 更新全局 min/max
        self.min_val = torch.min(current_min, self.min_val)
        self.max_val = torch.max(current_max, self.max_val)

        # print(f"  [Observer {id(self)}] Obs: min={current_min:.4f}, max={current_max:.4f} -> Global: min={self.min_val:.4f}, max={self.max_val:.4f}")

        return x_orig  # Observer 不改变数据流

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据收集到的 min/max 值计算量化参数 (scale, zero_point)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 计算得到的 scale 和 zero_point。
                                                返回的是标量张量。
        """
        # 获取量化范围 qmin, qmax (整数形式)
        qmin_raw, qmax_raw = calculate_qmin_qmax(self.dtype, self.reduce_range)
        # 将它们转换为浮点数以便计算
        qmin = float(qmin_raw)
        qmax = float(qmax_raw)

        device = self.min_val.device
        eps_tensor = torch.tensor(self.eps, device=device, dtype=torch.float32)

        # 1. 检查原始 min 和 max 是否相等
        if torch.isclose(self.min_val, self.max_val, atol=self.eps):
            # 如果原始 min/max 相等 (包括 == 0 和 != 0 的情况)
            # 统一使用 scale=1 作为约定
            scale = torch.tensor(1.0, device=device, dtype=torch.float32)
            # Zero point: 对称设为0，非对称设为qmin
            zero_point_val = 0.0 if self.qscheme == torch.per_tensor_symmetric else qmin
            zero_point = torch.tensor(
                int(round(zero_point_val)), dtype=torch.int64, device=device
            )

            # print(f"  [Observer {id(self)}] Calc (min==max): scale={scale:.4f}, zp={zero_point}")
            return scale, zero_point
        else:
            # 2. 标准计算路径 (min != max)

            if self.qscheme == torch.per_tensor_symmetric:
                # --- 对称量化计算 (Symmetric) ---
                # Zero point 强制为 0 (对于 qint 类型)
                zero_point = torch.tensor(0, dtype=torch.int64, device=device)

                # Scale 基于最大绝对值计算
                # 使用 *原始* 观测到的 min/max
                max_abs_val = torch.max(
                    torch.abs(self.min_val), torch.abs(self.max_val)
                )

                # 对称量化的有效量化范围上限 (如 qint8 是 127 或 63)
                # 注意：这里使用 qmax (已经考虑了 reduce_range)
                effective_qmax = qmax

                scale = max_abs_val / effective_qmax
                scale = torch.max(scale, eps_tensor)

                # print(f"  [Observer {id(self)}] Calc SYMMETRIC: max_abs={max_abs_val:.4f} => scale={scale:.4f}, zp={zero_point}")

            else:  # torch.per_tensor_affine
                # --- 非对称量化计算 (Asymmetric) ---
                # 确保 0 被包含在范围内
                zero_tensor = torch.tensor(0.0, device=device)
                min_val_adj = torch.min(zero_tensor, self.min_val)
                max_val_adj = torch.max(zero_tensor, self.max_val)

                # 标准非对称 scale
                scale = (max_val_adj - min_val_adj) / (qmax - qmin)
                scale = torch.max(scale, eps_tensor)

                # 标准非对称 zero_point (使用调整后的 min/max)
                zero_point = qmin - torch.round(min_val_adj / scale)

                # 最终将 zero_point 限制在量化范围内 (以防计算误差)
                # 注意：qmin_raw 和 qmax_raw 是整数类型
                zero_point = torch.clamp(zero_point, qmin_raw, qmax_raw)

                # print(f"  [Observer {id(self)}] Calc ASYMMETRIC: min={min_val_adj:.4f}, max={max_val_adj:.4f} => scale={scale:.4f}, zp={zero_point}")

            # 返回标量张量
            return scale.to(torch.float32), zero_point.to(torch.int64)

    def extra_repr(self) -> str:
        """为打印模块信息提供额外细节。"""
        return (
            f"dtype={self.dtype}, qscheme={self.qscheme}, reduce_range={self.reduce_range}, "
            f"min_val={self.min_val.item():.4f}, max_val={self.max_val.item():.4f}"
        )

    def reset_stats(self):
        """重置统计信息 (min_val, max_val)。"""
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        # print(f"  [Observer {id(self)}] 统计信息已重置")
