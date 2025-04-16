from typing import Tuple

import torch
import torch.nn as nn


# 定义量化范围辅助函数
def _calculate_qmin_qmax(dtype: torch.dtype) -> Tuple[int, int]:
    """根据数据类型计算量化范围 [qmin, qmax]。"""
    if dtype == torch.quint8:
        qmin, qmax = 0, 255
    elif dtype == torch.qint8:
        qmin, qmax = -128, 127
    elif dtype == torch.qint32:
        qmin, qmax = -2147483648, 2147483647
    # 可以根据需要添加更多类型，例如 float16 的模拟范围等
    else:
        raise ValueError(f"不支持的量化数据类型: {dtype}")
    return qmin, qmax


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
        # 获取量化范围 qmin, qmax
        qmin, qmax = _calculate_qmin_qmax(self.dtype)

        # 如果 reduce_range 为 True，调整 qmax (通常用于对称量化)
        if self.reduce_range:
            qmin, qmax = qmin // 2, qmax // 2

        # 处理 min_val 和 max_val 相等的情况（例如常数输入）
        min_val_actual = torch.min(
            self.min_val, torch.tensor(0.0, device=self.min_val.device)
        )
        max_val_actual = torch.max(
            self.max_val, torch.tensor(0.0, device=self.max_val.device)
        )

        # 如果 min == max，或者范围非常接近 0，则需要特殊处理以避免 scale 为 0 或 inf
        if torch.isclose(min_val_actual, max_val_actual, atol=1e-8):
            # 如果接近 0，设置一个小的默认范围（如 -1 到 1）
            if torch.isclose(
                min_val_actual,
                torch.tensor(0.0, device=min_val_actual.device),
                atol=1e-8,
            ):
                max_val_actual = min_val_actual + 1.0
            # 否则，稍微扩大范围
            else:
                scale = torch.tensor(1.0, device=min_val_actual.device)
                zero_point = torch.tensor(
                    0 if self.qscheme == torch.per_tensor_symmetric else qmin,
                    dtype=torch.int64,
                    device=min_val_actual.device,
                )
                # print(f"  [Observer {id(self)}] Calc (min==max): scale={scale:.4f}, zp={zero_point}")
                return scale, zero_point

        # --- 计算 Scale ---
        # (max - min) / (qmax - qmin)
        scale = (max_val_actual - min_val_actual) / float(qmax - qmin)
        # 确保 scale 不为 0 (加上 eps)
        scale = torch.max(scale, torch.tensor(self.eps, device=scale.device))

        # --- 计算 Zero Point ---
        if self.qscheme == torch.per_tensor_symmetric:
            # 对称量化：zero_point 通常为 0 (对于 qint8) 或 128 (对于 quint8，但不常见)
            # 这里对称量化时 zp 为 0
            zero_point = torch.tensor(0, dtype=torch.int64, device=scale.device)
        elif self.qscheme == torch.per_tensor_affine:
            # 非对称量化：zero_point = qmin - round(min_val / scale)
            # 使用浮点 zero_point 计算，然后四舍五入并 clamp 到 [qmin, qmax]
            zero_point_float = qmin - (min_val_actual / scale)
            zero_point = torch.round(zero_point_float).to(torch.int64)
            # Clamp zero_point 到有效范围
            zero_point = torch.clamp(zero_point, qmin, qmax)
        else:
            raise ValueError(f"不支持的 qscheme: {self.qscheme}")

        # print(f"  [Observer {id(self)}] Calc: min={self.min_val:.4f}, max={self.max_val:.4f} => scale={scale:.4f}, zp={zero_point}")

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
