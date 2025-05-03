# micronet/compression/fx/quantization/core/quant_utils.py

from typing import Tuple

import torch


# 定义量化范围辅助函数
def calculate_qmin_qmax(dtype: torch.dtype, reduce_range: bool) -> Tuple[int, int]:
    """
    根据数据类型和 reduce_range 参数计算量化范围 [qmin, qmax]。
    """
    if dtype == torch.quint8:
        qmin, qmax = 0, 255
        if reduce_range:
            # PyTorch quint8 reduce_range: [0, 127]
            qmax = 127
    elif dtype == torch.qint8:
        qmin, qmax = -128, 127
        if reduce_range:
            # PyTorch qint8 reduce_range: [-127, 127]
            qmin = -127
    elif dtype == torch.qint32:
        # qint32 通常不使用 reduce_range，但以防万一
        qmin, qmax = -2147483648, 2147483647
        if reduce_range:
            # PyTorch qint32 reduce_range 行为未明确定义，通常不调整
            pass  # 或者可以实现类似的 qmin = -2147483647
    else:
        raise ValueError(f"不支持的量化数据类型: {dtype}")

    return qmin, qmax
