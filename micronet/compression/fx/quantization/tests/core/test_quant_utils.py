# micronet/compression/fx/quantization/tests/test_quant_utils.py

import pytest
import torch

from micronet.compression.fx.quantization.core.quant_utils import calculate_qmin_qmax

# --- 测试 calculate_qmin_qmax 函数 ---

# 1. 定义有效的测试用例参数
# (dtype, reduce_range, expected_qmin, expected_qmax)
valid_qmin_qmax_test_cases = [
    # --- torch.quint8 ---
    (torch.quint8, False, 0, 255),  # 标准范围
    (torch.quint8, True, 0, 127),  # 缩减范围
    # --- torch.qint8 ---
    (torch.qint8, False, -128, 127),  # 标准范围
    (torch.qint8, True, -127, 127),  # 缩减范围 (PyTorch 标准)
    # --- torch.qint32 ---
    (torch.qint32, False, -2147483648, 2147483647),  # 标准范围
    (torch.qint32, True, -2147483648, 2147483647),  # 缩减范围 (当前实现不改变)
]


@pytest.mark.parametrize(
    "dtype, reduce_range, expected_qmin, expected_qmax", valid_qmin_qmax_test_cases
)
def test_calculate_qmin_qmax_valid(dtype, reduce_range, expected_qmin, expected_qmax):
    """
    测试 calculate_qmin_qmax 在有效的数据类型和 reduce_range 组合下是否返回正确的 qmin, qmax。
    """
    # 对于每组有效的输入 (数据类型, 是否缩减范围)，
    # 调用函数 calculate_qmin_qmax，
    # 然后断言返回的 (最小值, 最大值) 元组是否与预期的 (最小值, 最大值) 完全相等。

    qmin, qmax = calculate_qmin_qmax(dtype, reduce_range)
    assert (qmin, qmax) == (
        expected_qmin,
        expected_qmax,
    ), f"对于 dtype={dtype}, reduce_range={reduce_range}，期望得到 ({expected_qmin}, {expected_qmax})，但实际得到 ({qmin}, {qmax})"


# 2. 定义无效的数据类型参数
invalid_dtype_test_cases = [
    (torch.float32, False),
    (torch.float16, False),
    (torch.float64, False),
    (torch.int16, False),
    (torch.int64, False),
    (torch.bool, False),
    (torch.float32, True),  # 也测试 reduce_range=True 的情况
    (torch.int16, True),
]


@pytest.mark.parametrize("invalid_dtype, reduce_range", invalid_dtype_test_cases)
def test_calculate_qmin_qmax_invalid_dtype(invalid_dtype, reduce_range):
    """
    测试 calculate_qmin_qmax 在接收到不支持的数据类型时是否正确抛出 ValueError。
    """
    # 对于每组无效的输入 (不支持的数据类型, 是否缩减范围)，
    # 使用 pytest.raises 上下文管理器来捕获预期的 ValueError。
    # 断言：调用 calculate_qmin_qmax 必须抛出 ValueError。
    # 进一步断言：抛出的 ValueError 的错误信息必须包含 "不支持的量化数据类型"。

    expected_error_message = "不支持的量化数据类型"
    with pytest.raises(ValueError, match=expected_error_message):
        calculate_qmin_qmax(invalid_dtype, reduce_range)
