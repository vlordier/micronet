# micronet/compression/fx/quantization/tests/core/test_observer.py

import pytest
import torch

from micronet.compression.fx.quantization.core.observer import MinMaxObserver
from micronet.compression.fx.quantization.core.quant_utils import calculate_qmin_qmax

# --- 测试辅助函数 _calculate_qmin_qmax ---


@pytest.mark.parametrize(
    "dtype, reduce_range, expected_qmin, expected_qmax",
    [
        # reduce_range = False
        (torch.quint8, False, 0, 255),
        (torch.qint8, False, -128, 127),
        (torch.qint32, False, -2147483648, 2147483647),
        # reduce_range = True
        (torch.quint8, True, 0, 127),  # 0 // 2 = 0, 255 // 2 = 127
        (torch.qint8, True, -127, 127),  # -128 // 2 = -64, 127 // 2 = 63
        (torch.qint32, True, -2147483648, 2147483647),  # -2^31 // 2, (2^31 - 1) // 2
    ],
)
def test_calculate_qmin_qmax_supported(
    dtype, reduce_range, expected_qmin, expected_qmax
):
    """测试 calculate_qmin_qmax 对支持的数据类型和 reduce_range 选项"""
    qmin, qmax = calculate_qmin_qmax(dtype, reduce_range)
    assert qmin == expected_qmin
    assert qmax == expected_qmax


def test_calculate_qmin_qmax_unsupported():
    """测试 _calculate_qmin_qmax 对不支持的数据类型"""
    # 调用时传入 reduce_range=False (或 True，结果应相同，因为错误发生在之前)
    with pytest.raises(ValueError, match="不支持的量化数据类型"):
        calculate_qmin_qmax(torch.float32, False)
    with pytest.raises(ValueError, match="不支持的量化数据类型"):
        calculate_qmin_qmax(torch.int16, False)
    # 可以选择性地再测试 reduce_range=True 的情况，虽然逻辑上没必要
    with pytest.raises(ValueError, match="不支持的量化数据类型"):
        calculate_qmin_qmax(torch.float64, True)


# --- 测试 MinMaxObserver 类 ---


@pytest.fixture
def default_observer() -> MinMaxObserver:
    """提供一个默认的 MinMaxObserver 实例"""
    return MinMaxObserver()


@pytest.fixture
def qint8_sym_observer() -> MinMaxObserver:
    """提供一个 qint8 对称量化 Observer 实例"""
    return MinMaxObserver(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)


def test_observer_init_defaults(default_observer: MinMaxObserver):
    """测试 Observer 的默认初始化"""
    assert default_observer.dtype == torch.quint8
    assert default_observer.qscheme == torch.per_tensor_affine
    assert default_observer.reduce_range is False
    assert default_observer.min_val == float("inf")
    assert default_observer.max_val == float("-inf")
    assert "min_val" in default_observer._buffers
    assert "max_val" in default_observer._buffers


def test_observer_init_custom():
    """测试 Observer 的自定义初始化"""
    obs = MinMaxObserver(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=True,
        eps=1e-6,
    )
    assert obs.dtype == torch.qint8
    assert obs.qscheme == torch.per_tensor_symmetric
    assert obs.reduce_range is True
    assert obs.eps == 1e-6
    assert obs.min_val == float("inf")
    assert obs.max_val == float("-inf")


def test_observer_forward_updates_stats(default_observer: MinMaxObserver):
    """测试 forward 方法是否正确更新 min/max 统计值"""
    t1 = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    t2 = torch.tensor([-2.0, -1.5, 0.5, 3.0])

    out1 = default_observer(t1)
    assert torch.equal(out1, t1)  # forward 不应修改输入
    assert default_observer.min_val == -1.0
    assert default_observer.max_val == 2.0

    out2 = default_observer(t2)
    assert torch.equal(out2, t2)
    assert default_observer.min_val == -2.0  # 更新了全局 min
    assert default_observer.max_val == 3.0  # 更新了全局 max


def test_observer_forward_empty_tensor(default_observer: MinMaxObserver):
    """测试 forward 方法处理空张量"""
    empty_tensor = torch.tensor([])
    out = default_observer(empty_tensor)
    assert torch.equal(out, empty_tensor)
    # min/max 应该保持初始状态
    assert default_observer.min_val == float("inf")
    assert default_observer.max_val == float("-inf")


# --- 测试 calculate_qparams ---

# 测试 calculate_qparams 的各种情况
# (dtype, qscheme, reduce_range, min_in, max_in, expected_scale, expected_zp)
qparam_test_cases = [
    # --- 对称 qint8 ---
    # reduce_range=False (qmax=127.0)
    (torch.qint8, torch.per_tensor_symmetric, False, -1.0, 1.0, 1.0 / 127.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, False, -5.0, 3.0, 5.0 / 127.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, False, -2.5, 8.0, 8.0 / 127.0, 0),
    # reduce_range=True (qmax=127.0)
    (torch.qint8, torch.per_tensor_symmetric, True, -1.0, 1.0, 1.0 / 127.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, True, -5.0, 3.0, 5.0 / 127.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, True, -2.5, 8.0, 8.0 / 127.0, 0),
    # --- 非对称 quint8 ---
    # reduce_range=False (qmin=0.0, qmax=255.0)
    (torch.quint8, torch.per_tensor_affine, False, 0.0, 1.0, 1.0 / 255.0, 0),
    (torch.quint8, torch.per_tensor_affine, False, 1.0, 5.0, 5.0 / 255.0, 0),
    (
        torch.quint8,
        torch.per_tensor_affine,
        False,
        -1.0,
        5.0,
        6.0 / 255.0,
        42,
    ),  # min_adj=-1, max_adj=5, scale=6/255, zp=0-round(-1/scale)=42
    # reduce_range=True (qmin=0.0, qmax=127.0, 因为 255 // 2 = 127)
    (torch.quint8, torch.per_tensor_affine, True, 0.0, 1.0, 1.0 / 127.0, 0),
    (torch.quint8, torch.per_tensor_affine, True, 1.0, 5.0, 5.0 / 127.0, 0),
    (
        torch.quint8,
        torch.per_tensor_affine,
        True,
        -1.0,
        5.0,
        6.0 / 127.0,
        21,
    ),  # min_adj=-1, max_adj=5, scale=6/127, zp=0-round(-1/scale)=21
    # --- 非对称 qint8 ---
    # reduce_range=False (qmin=-128.0, qmax=127.0)
    (torch.qint8, torch.per_tensor_affine, False, -1.0, 1.0, 2.0 / 255.0, -1),
    (torch.qint8, torch.per_tensor_affine, False, -8.0, 8.0, 16.0 / 255.0, -1),
    (
        torch.qint8,
        torch.per_tensor_affine,
        False,
        0.0,
        10.0,
        10.0 / 255.0,
        -128,
    ),  # 仅正范围
    (
        torch.qint8,
        torch.per_tensor_affine,
        False,
        -5.0,
        0.0,
        5.0 / 255.0,
        127,
    ),  # 仅负范围
    # reduce_range=True (qmin=-127.0, qmax=127.0)
    (
        torch.qint8,
        torch.per_tensor_affine,
        True,
        -1.0,
        1.0,
        2.0 / 254.0,
        0,
    ),
    (
        torch.qint8,
        torch.per_tensor_affine,
        True,
        0.0,
        5.0,
        5.0 / 254.0,
        -127,
    ),
    (
        torch.qint8,
        torch.per_tensor_affine,
        True,
        -8.0,
        0.0,
        8.0 / 254.0,
        127,
    ),
]


@pytest.mark.parametrize(
    "dtype, qscheme, reduce_range, min_in, max_in, expected_scale, expected_zp",
    qparam_test_cases,
)
def test_observer_calculate_qparams_various(
    dtype, qscheme, reduce_range, min_in, max_in, expected_scale, expected_zp
):
    """测试 calculate_qparams 在各种情况下的计算"""
    obs = MinMaxObserver(dtype=dtype, qscheme=qscheme, reduce_range=reduce_range)
    # 手动设置 min/max
    obs.min_val.fill_(min_in)
    obs.max_val.fill_(max_in)

    scale, zero_point = obs.calculate_qparams()

    assert isinstance(scale, torch.Tensor)
    assert scale.dtype == torch.float32
    assert scale.shape == ()  # 标量
    assert isinstance(zero_point, torch.Tensor)
    assert zero_point.dtype == torch.int64
    assert zero_point.shape == ()  # 标量

    # 使用 pytest.approx 进行浮点比较
    assert scale.item() == pytest.approx(expected_scale)
    assert zero_point.item() == expected_zp


# 测试 calculate_qparams 当 min == max
# (dtype, qscheme, min_max_val, expected_scale, expected_zp)
# 注意：当 min == max 时，scale 预期为 1.0
qparam_min_max_equal_cases = [
    # 无符号 (quint8), qmin=0
    (torch.quint8, torch.per_tensor_affine, 0.0, 1.0, 0),  # 非对称, min=max=0 -> qmin=0
    (
        torch.quint8,
        torch.per_tensor_affine,
        5.0,
        1.0,
        0,
    ),  # 非对称, min=max!=0 -> qmin=0
    (
        torch.quint8,
        torch.per_tensor_affine,
        -3.0,
        1.0,
        0,
    ),  # 非对称, min=max!=0 -> qmin=0
    # 有符号 (qint8), qmin=-128
    (
        torch.qint8,
        torch.per_tensor_affine,
        0.0,
        1.0,
        -128,
    ),  # 非对称, min=max=0 -> qmin=-128
    (
        torch.qint8,
        torch.per_tensor_affine,
        5.0,
        1.0,
        -128,
    ),  # 非对称, min=max!=0 -> qmin=-128
    (
        torch.qint8,
        torch.per_tensor_affine,
        -3.0,
        1.0,
        -128,
    ),  # 非对称, min=max!=0 -> qmin=-128
    # 对称 (qint8) - zero_point 必须是 0
    (torch.qint8, torch.per_tensor_symmetric, 0.0, 1.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, 5.0, 1.0, 0),
    (torch.qint8, torch.per_tensor_symmetric, -3.0, 1.0, 0),
]


@pytest.mark.parametrize(
    "dtype, qscheme, min_max_val, expected_scale, expected_zp",
    qparam_min_max_equal_cases,
)
def test_observer_calculate_qparams_min_max_equal(
    dtype, qscheme, min_max_val, expected_scale, expected_zp
):
    """测试 calculate_qparams 当 min == max 时的特殊处理"""
    obs = MinMaxObserver(dtype=dtype, qscheme=qscheme)
    obs.min_val.fill_(min_max_val)
    obs.max_val.fill_(min_max_val)

    scale, zero_point = obs.calculate_qparams()

    assert scale.item() == pytest.approx(expected_scale)
    assert zero_point.item() == expected_zp  # 直接比较整数 zp


def test_observer_calculate_qparams_initial_state(default_observer: MinMaxObserver):
    """测试在初始状态 (min=inf, max=-inf) 调用 calculate_qparams"""
    # --- 预期行为 ---
    # isclose(inf, -inf) is False -> 进入 else 块 (非对称)
    # min_val_adj = min(inf, 0) = 0
    # max_val_adj = max(-inf, 0) = 0
    # scale = max( (0-0)/255, eps ) = eps
    # zero_point = qmin - round(min_val_adj / scale)
    #            = 0 - round(0 / eps) = 0 - 0 = 0
    # clamp(0, 0, 255) = 0
    # ---------------------------------------------
    expected_scale = default_observer.eps
    expected_zp = 0

    # 如果观察者是对称的，预期 zp 仍为 0
    if default_observer.qscheme == torch.per_tensor_symmetric:
        expected_zp = 0  # 对于对称量化, 初始 ZP 也是 0

    scale, zero_point = default_observer.calculate_qparams()

    print(f"\nInitial State Test:")
    print(
        f"  Observer Config: dtype={default_observer.dtype}, scheme={default_observer.qscheme}"
    )
    print(f"  Expected scale: {expected_scale:.8f}, zp: {expected_zp}")
    print(f"  Actual scale:   {scale.item():.8f}, zp: {zero_point.item()}")

    assert scale.item() == pytest.approx(expected_scale)
    assert zero_point.item() == expected_zp


def test_observer_reset_stats(default_observer: MinMaxObserver):
    """测试 reset_stats 方法"""
    # 先更新一些统计值
    default_observer(torch.tensor([-1.0, 2.0]))
    assert default_observer.min_val != float("inf")
    assert default_observer.max_val != float("-inf")

    # 重置
    default_observer.reset_stats()

    # 检查是否恢复到初始状态
    assert default_observer.min_val == float("inf")
    assert default_observer.max_val == float("-inf")


def test_observer_extra_repr(default_observer: MinMaxObserver):
    """测试 extra_repr 方法的输出"""
    default_observer(torch.tensor([-0.5, 1.5]))
    repr_str = default_observer.extra_repr()
    assert isinstance(repr_str, str)
    assert "dtype=torch.quint8" in repr_str
    assert "qscheme=torch.per_tensor_affine" in repr_str
    assert "reduce_range=False" in repr_str
    assert "min_val=-0.5000" in repr_str  # 检查格式
    assert "max_val=1.5000" in repr_str


def test_observer_state_dict(default_observer: MinMaxObserver):
    """测试 state_dict 的保存和加载"""
    t = torch.tensor([-3.0, 5.0])
    default_observer(t)
    state = default_observer.state_dict()

    # 检查 state_dict 包含 buffers
    assert "min_val" in state
    assert "max_val" in state
    assert state["min_val"] == -3.0
    assert state["max_val"] == 5.0

    # 加载到新的 observer
    new_observer = MinMaxObserver()
    new_observer.load_state_dict(state)

    assert new_observer.min_val == -3.0
    assert new_observer.max_val == 5.0
    assert new_observer.dtype == default_observer.dtype  # 确保其他属性没变
    assert new_observer.qscheme == default_observer.qscheme
