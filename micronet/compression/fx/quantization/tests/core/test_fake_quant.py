# micronet/compression/fx/quantization/tests/core/test_fake_quant.py

import pytest
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Type
import copy
import io

from micronet.compression.fx.quantization.core.fake_quant import (
    FakeQuantize,
    QATMode,
    MinMaxObserver,
)
from micronet.compression.fx.quantization.core.quant_utils import calculate_qmin_qmax

# --- 设备 Fixture ---
_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    _DEVICES.append(torch.device("cuda:0"))


@pytest.fixture(params=_DEVICES)
def device(request):
    """提供可用的计算设备 (CPU, 或 CUDA 如果可用)"""
    return request.param


# --- 基础 Fixtures ---


@pytest.fixture
def default_observer_cls() -> Type[nn.Module]:
    """提供一个默认的 Observer 类 (MinMaxObserver)"""
    return MinMaxObserver


@pytest.fixture
def default_fake_quant(default_observer_cls, device) -> FakeQuantize:
    """提供一个具有默认配置的 FakeQuantize 实例，并移动到指定设备"""
    return FakeQuantize(observer_cls=default_observer_cls).to(device)


@pytest.fixture
def qat_learning_fake_quant(default_observer_cls, device) -> FakeQuantize:
    """提供一个为 QAT LEARNING_ONLY 模式配置好的 FakeQuantize 实例"""
    fq = FakeQuantize(
        observer_cls=default_observer_cls,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    ).to(
        device
    )  # 移动到设备
    fq.enable_qat(enabled=True, qat_mode=QATMode.LEARNING_ONLY, inherit_qparams=False)
    # 手动设置一个初始 scale/zp
    with torch.no_grad():
        fq.scale.fill_(0.1)
        fq.zero_point.fill_(0.0)  # 对称量化 zp=0, 即使设了初始值也会被 learning 覆盖
    fq.scale.requires_grad_(True)  # 确认梯度开启
    fq.zero_point.requires_grad_(True)  # 确认梯度开启
    return fq


@pytest.fixture
def qat_hybrid_fake_quant(default_observer_cls, device) -> FakeQuantize:
    """提供一个为 QAT HYBRID_EMA 模式配置好的 FakeQuantize 实例"""
    fq = FakeQuantize(
        observer_cls=default_observer_cls,
        dtype=torch.quint8,
        reduce_range=False,
    ).to(device)
    fq.enable_qat(
        enabled=True, qat_mode=QATMode.HYBRID_EMA, inherit_qparams=False, ema_alpha=0.9
    )
    # 手动设置一个初始 scale/zp
    with torch.no_grad():
        fq.scale.fill_(0.05)
        fq.zero_point.fill_(120.5)
    fq.scale.requires_grad_(True)
    fq.zero_point.requires_grad_(True)
    fq.enable_observer(True)  # 确保 observer 开启
    return fq


# --- 测试类 ---


class TestFakeQuantInit:
    """测试 FakeQuantize 的初始化逻辑"""

    def test_initialization_defaults(self, default_fake_quant: FakeQuantize, device):
        """验证：默认初始化时，属性是否符合预期，参数是否在正确设备"""
        fq = default_fake_quant
        assert isinstance(fq.observer, MinMaxObserver)
        assert fq.observer_cls is MinMaxObserver
        assert fq.dtype == torch.quint8
        assert fq.qscheme == torch.per_tensor_affine
        assert fq.reduce_range is False
        assert fq.observer.reduce_range is False
        qmin, qmax = calculate_qmin_qmax(torch.quint8, False)
        assert fq.quant_min == qmin
        assert fq.quant_max == qmax
        assert isinstance(fq.scale, Parameter)
        assert isinstance(fq.zero_point, Parameter)
        assert fq.scale.dtype == torch.float32
        assert fq.zero_point.dtype == torch.float32
        assert fq.observer_enabled is True
        assert fq.fake_quant_enabled is False
        assert fq.is_qat_learning is False
        assert fq.qat_mode is None
        assert fq.scale.requires_grad is False
        assert fq.zero_point.requires_grad is False
        # 检查设备
        assert fq.scale.device == device
        assert fq.zero_point.device == device
        assert fq.observer.min_val.device == device
        assert fq.observer.max_val.device == device

    def test_initialization_custom(self, default_observer_cls, device):
        """验证：自定义初始化，属性设置及设备"""
        fq = FakeQuantize(
            observer_cls=default_observer_cls,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=True,
            initial_scale=0.5,
            initial_zero_point=-5,  # 对称量化 zp 通常为0，但初始化应能接受
            eps=1e-6,
        ).to(device)
        assert fq.dtype == torch.qint8
        assert fq.qscheme == torch.per_tensor_symmetric
        assert fq.reduce_range is True
        assert fq.observer.reduce_range is True
        qmin, qmax = calculate_qmin_qmax(torch.qint8, True)
        assert fq.quant_min == qmin
        assert fq.quant_max == qmax
        assert fq.scale.item() == 0.5
        # 注意: QAT 对称模式可能会在学习中强制 zero_point=0，但初始化应存储值
        assert fq.zero_point.item() == -5.0
        assert fq.eps == 1e-6
        # 检查设备
        assert fq.scale.device == device
        assert fq.zero_point.device == device


class TestFakeQuantStateManagement:
    """测试 FakeQuantize 的状态管理方法"""

    def test_enable_observer(self, default_fake_quant: FakeQuantize):
        """验证：enable_observer 方法是否能正确启用/禁用 observer_enabled 标志"""
        fq = default_fake_quant
        fq.enable_observer(True)
        assert fq.observer_enabled is True
        fq.enable_observer(False)
        assert fq.observer_enabled is False
        # 在 QAT STATS_ONLY/HYBRID_EMA 模式下禁用 observer 应报错
        fq.enable_qat(True, qat_mode=QATMode.HYBRID_EMA)
        with pytest.raises(
            ValueError, match=f"禁用 Observer 与 {fq.qat_mode.name} QAT 模式不兼容"
        ):
            fq.enable_observer(False)

    def test_enable_fake_quant(self, default_fake_quant: FakeQuantize):
        """验证：enable_fake_quant 方法是否能正确启用/禁用 fake_quant_enabled 标志"""
        fq = default_fake_quant
        fq.enable_fake_quant(True)
        assert fq.fake_quant_enabled is True
        fq.enable_fake_quant(False)
        assert fq.fake_quant_enabled is False

    def test_enable_qat_learning_only(self, default_fake_quant: FakeQuantize):
        """验证：启用 QAT 并设置为 LEARNING_ONLY 模式时的状态变化"""
        fq = default_fake_quant
        # 记录初始 observer 状态
        initial_observer_state = fq.observer_enabled

        fq.enable_qat(True, qat_mode=QATMode.LEARNING_ONLY, inherit_qparams=False)
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.LEARNING_ONLY
        assert fq.fake_quant_enabled is True
        # 在 LEARNING_ONLY 模式下，observer 默认会被关闭 (除非之前就是关的)
        if initial_observer_state:
            assert fq.observer_enabled is False
        else:
            assert fq.observer_enabled is False  # 保持关闭
        assert fq.scale.requires_grad is True
        assert fq.zero_point.requires_grad is True

        # 测试退出 QAT
        fq.enable_qat(False)
        assert fq.is_qat_learning is False
        assert fq.qat_mode is None
        assert fq.scale.requires_grad is False
        assert fq.zero_point.requires_grad is False
        assert fq.observer_enabled is False  # 退出 QAT 时 observer 通常关闭
        assert fq.fake_quant_enabled is True  # 退出 QAT 时 fake_quant 保持开启

    def test_enable_qat_stats_only(self, default_fake_quant: FakeQuantize):
        """验证：启用 QAT 并设置为 STATS_ONLY 模式时的状态变化"""
        fq = default_fake_quant
        fq.enable_qat(True, qat_mode=QATMode.STATS_ONLY)
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.STATS_ONLY
        assert fq.fake_quant_enabled is True
        assert fq.observer_enabled is True
        assert fq.scale.requires_grad is False
        assert fq.zero_point.requires_grad is False

    def test_enable_qat_hybrid_ema(self, default_fake_quant: FakeQuantize):
        """验证：启用 QAT 并设置为 HYBRID_EMA 模式时的状态变化"""
        fq = default_fake_quant
        fq.enable_qat(True, qat_mode=QATMode.HYBRID_EMA, ema_alpha=0.8)
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.HYBRID_EMA
        assert fq.fake_quant_enabled is True
        assert fq.observer_enabled is True
        assert fq.ema_alpha == 0.8
        assert fq.scale.requires_grad is True
        assert fq.zero_point.requires_grad is True
        # 测试无效 ema_alpha
        with pytest.raises(
            ValueError, match=r"HYBRID_EMA 模式需要有效的 ema_alpha \(0 <= alpha <= 1\)"
        ):
            fq.enable_qat(True, qat_mode=QATMode.HYBRID_EMA, ema_alpha=1.1)

    def test_inherit_qparams(self, default_fake_quant: FakeQuantize, device):
        """验证：启用 QAT 时，inherit_qparams=True 能否从 observer 继承参数"""
        fq = default_fake_quant
        # 先让 observer 收集一些统计数据 (确保数据在设备上)
        fq.observer(torch.tensor([-1.0, 0.0, 3.0], device=device))
        expected_scale, expected_zp = fq.observer.calculate_qparams()
        initial_scale = fq.scale.item()
        initial_zp = fq.zero_point.item()

        fq.enable_qat(True, qat_mode=QATMode.LEARNING_ONLY, inherit_qparams=True)

        assert fq.scale.item() == pytest.approx(expected_scale.item())
        assert fq.zero_point.item() == pytest.approx(expected_zp.item())
        assert fq.scale.item() != initial_scale  # 确认值被更新
        assert fq.zero_point.item() != initial_zp


class TestFakeQuantObserverInteraction:
    """测试 FakeQuantize 与内部 Observer 的交互"""

    def test_reset_observer_stats(self, default_fake_quant: FakeQuantize, device):
        """验证：调用 reset_observer_stats 是否能重置 observer 的 min/max 值"""
        fq = default_fake_quant
        fq.observer(torch.tensor([-1.0, 1.0], device=device))
        assert fq.observer.min_val.item() == -1.0
        assert fq.observer.max_val.item() == 1.0

        fq.reset_observer_stats()

        assert torch.isinf(fq.observer.min_val) and fq.observer.min_val > 0
        assert torch.isinf(fq.observer.max_val) and fq.observer.max_val < 0

    def test_calculate_qparams_updates_params(
        self, default_fake_quant: FakeQuantize, device
    ):
        """验证：调用 calculate_qparams 是否会使用 observer 的统计结果更新模块的 scale/zp"""
        fq = default_fake_quant
        fq.observer(torch.tensor([0.0, 2.0], device=device))
        expected_scale, expected_zp = fq.observer.calculate_qparams()

        # calculate_qparams 应该直接更新参数，不需要先 enable_fake_quant
        s_updated, zp_updated = fq.calculate_qparams()

        assert fq.scale.item() == pytest.approx(expected_scale.item())
        assert fq.zero_point.item() == pytest.approx(expected_zp.item())
        assert torch.equal(s_updated, fq.scale)  # 验证返回值
        assert torch.equal(zp_updated, fq.zero_point)

    def test_calculate_qparams_on_device(
        self, default_fake_quant: FakeQuantize, device
    ):
        """验证：calculate_qparams 返回的 tensor 在正确设备上"""
        fq = default_fake_quant
        fq.observer(torch.tensor([0.0, 2.0], device=device))  # 输入在设备上
        s, zp = fq.calculate_qparams()
        assert s.device == device
        assert zp.device == device  # zero_point 是 Parameter，应在设备上

    def test_ema_update(self, qat_hybrid_fake_quant: FakeQuantize, device):
        """验证：在 HYBRID_EMA 模式下，forward 是否会触发 EMA 更新"""
        fq = qat_hybrid_fake_quant
        initial_scale = fq.scale.clone()
        initial_zp = fq.zero_point.clone()

        # 让 observer 看到不同的数据
        fq.observer(torch.tensor([-5.0, 5.0], device=device))
        stat_scale, stat_zp = fq.observer.calculate_qparams()

        # 执行 forward
        x = torch.randn(10, device=device)
        _ = fq(x)

        # 检查 scale 和 zero_point 是否被 EMA 更新
        expected_scale = fq.ema_alpha * initial_scale + (1 - fq.ema_alpha) * stat_scale
        expected_zp = fq.ema_alpha * initial_zp + (1 - fq.ema_alpha) * stat_zp.float()

        assert fq.scale.item() == pytest.approx(expected_scale.item())
        assert fq.zero_point.item() == pytest.approx(expected_zp.item())


class TestFakeQuantForward:
    """测试 FakeQuantize 的 forward 方法 (包括边缘情况和设备)"""

    def test_forward_observer_only(self, default_fake_quant: FakeQuantize, device):
        """验证：当只有 observer 启用时，forward 收集统计信息但不修改输入"""
        fq = default_fake_quant
        fq.enable_observer(True)
        fq.enable_fake_quant(False)
        x = torch.tensor([-1.0, 0.0, 3.0], device=device)
        y = fq(x)
        assert torch.equal(x, y)
        assert fq.observer.min_val.item() == pytest.approx(-1.0)
        assert fq.observer.max_val.item() == pytest.approx(3.0)

    def test_forward_fake_quant_only(self, default_fake_quant: FakeQuantize, device):
        """验证：当只有 fake quant 启用时 (PTQ 推理模式)，forward 执行伪量化"""
        fq = default_fake_quant
        fq.enable_observer(False)
        fq.enable_fake_quant(True)
        # 手动设置 scale/zp
        with torch.no_grad():
            fq.scale.fill_(0.1)
            fq.zero_point.fill_(128.0)  # quint8 的零点
        x = torch.tensor([-0.51, 0.0, 0.51, 1.0], device=device)  # 使用不易精确表示的值

        # 手动计算预期输出 (根据 FakeQuantize 的实现)
        # x_q = round(x / (scale + eps) + zp)
        # x_q = round([-5.1, 0, 5.1, 10] + 128) = round([122.9, 128, 133.1, 138]) = [123, 128, 133, 138]
        # x_q_clamped = clamp([123, 128, 133, 138], 0, 255) = [123, 128, 133, 138]
        # x_dq = (x_q_clamped - zp) * scale = [-5, 0, 5, 10] * 0.1 = [-0.5, 0.0, 0.5, 1.0]
        x_dq_manual = torch.tensor([-0.5, 0.0, 0.5, 1.0], device=device)
        # STE output = x + (x_dq - x).detach()
        expected_y = x + (x_dq_manual - x).detach()

        y = fq(x)

        assert torch.allclose(
            y, expected_y, atol=1e-6
        ), f"Expected STE output {expected_y}, got {y}"
        # 验证 observer 未运行
        assert torch.isinf(fq.observer.min_val)

    def test_forward_qat_learning_only(
        self, qat_learning_fake_quant: FakeQuantize, device
    ):
        """验证：在 QAT LEARNING_ONLY 模式下，forward 执行伪量化，参数可学习，observer 关闭"""
        fq = qat_learning_fake_quant
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.LEARNING_ONLY
        assert fq.observer_enabled is False
        assert fq.fake_quant_enabled is True
        assert fq.scale.requires_grad is True
        assert fq.zero_point.requires_grad is True

        x = torch.randn(5, device=device, requires_grad=True) * 5
        y = fq(x)

        assert not torch.equal(x, y), "QAT forward output should differ from input"
        assert torch.isinf(
            fq.observer.min_val
        ), "Observer should not run in LEARNING_ONLY"

    def test_forward_qat_stats_only(self, default_observer_cls, device):
        """验证：在 QAT STATS_ONLY 模式下，forward 收集统计并更新参数，参数不可学习"""
        fq = FakeQuantize(
            observer_cls=default_observer_cls,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ).to(device)
        fq.enable_qat(True, qat_mode=QATMode.STATS_ONLY)
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.STATS_ONLY
        assert fq.observer_enabled is True
        assert fq.fake_quant_enabled is True
        assert fq.scale.requires_grad is False
        assert fq.zero_point.requires_grad is False

        x = torch.tensor([-1.5, 0.1, 3.8], device=device)
        y = fq(x)

        # 检查 observer 运行
        assert fq.observer.min_val.item() == pytest.approx(-1.5)
        assert fq.observer.max_val.item() == pytest.approx(3.8)

        # 检查 scale/zp 被更新
        expected_scale, expected_zp = (
            fq.observer.calculate_qparams()
        )  # 重新计算以确保最新
        assert fq.scale.item() == pytest.approx(expected_scale.item())
        assert fq.zero_point.item() == pytest.approx(expected_zp.item())

        assert not torch.equal(
            x, y
        ), "STATS_ONLY forward output should differ from input"

    def test_forward_qat_hybrid_ema(self, qat_hybrid_fake_quant: FakeQuantize, device):
        """验证：在 QAT HYBRID_EMA 模式下，forward 收集统计、EMA更新参数、参数可学习"""
        fq = qat_hybrid_fake_quant
        assert fq.is_qat_learning is True
        assert fq.qat_mode == QATMode.HYBRID_EMA
        assert fq.observer_enabled is True
        assert fq.fake_quant_enabled is True
        assert fq.scale.requires_grad is True
        assert fq.zero_point.requires_grad is True

        initial_scale = fq.scale.clone()
        initial_zp = fq.zero_point.clone()
        x = torch.tensor([-1.0, 1.0, 2.0], device=device)
        y = fq(x)

        assert fq.observer.min_val.item() == pytest.approx(-1.0)
        assert fq.observer.max_val.item() == pytest.approx(2.0)
        # 检查 EMA 更新发生 (与 test_ema_update 逻辑相同)
        stat_scale, stat_zp = fq.observer.calculate_qparams()
        expected_scale = fq.ema_alpha * initial_scale + (1 - fq.ema_alpha) * stat_scale
        expected_zp = fq.ema_alpha * initial_zp + (1 - fq.ema_alpha) * stat_zp.float()
        assert fq.scale.item() == pytest.approx(expected_scale.item())
        assert fq.zero_point.item() == pytest.approx(expected_zp.item())

        assert not torch.equal(
            x, y
        ), "HYBRID_EMA forward output should differ from input"

    def test_forward_reduce_range(self, default_observer_cls, device):
        """明确验证 reduce_range=True 的效果"""
        fq = FakeQuantize(
            observer_cls=default_observer_cls,
            dtype=torch.qint8,  # [-128, 127]
            reduce_range=True,  # qint8 reduce range -> [-127, 127] in PyTorch
            initial_scale=0.1,
            initial_zero_point=0.0,  # 对称 qint8
        ).to(device)
        fq.enable_fake_quant(True)
        fq.enable_observer(False)

        assert fq.quant_min == -127, "qint8 with reduce_range should have qmin=-127"
        assert fq.quant_max == 127, "qint8 with reduce_range should have qmax=127"

        x = torch.tensor([-15.0, 0.0, 15.0], device=device)
        # 手动计算:
        # scale=0.1, zp=0.0
        # x_q = round(x / 0.1 + 0) = round([-150, 0, 150]) = [-150, 0, 150]
        # x_q_clamped = clamp([-150, 0, 150], -127, 127) = [-127, 0, 127]
        # x_dq = (x_q_clamped - 0) * 0.1 = [-12.7, 0.0, 12.7]
        x_dq_manual = torch.tensor([-12.7, 0.0, 12.7], device=device)
        expected_y = x + (x_dq_manual - x).detach()  # STE
        y = fq(x)
        assert torch.allclose(
            y, expected_y, atol=1e-6
        ), f"reduce_range: Expected {expected_y}, got {y}"

    def test_forward_empty_input(self, default_fake_quant: FakeQuantize, device):
        """验证：输入为空张量时的行为"""
        fq = default_fake_quant
        fq.enable_fake_quant(True)
        x_empty = torch.tensor([], device=device)
        y = fq(x_empty)
        assert y.shape == x_empty.shape
        assert y.dtype == x_empty.dtype
        assert y.device == device
        # 检查 observer (如果启用)
        fq.enable_observer(True)
        y_obs = fq(x_empty)
        assert torch.isinf(fq.observer.min_val)
        assert torch.isinf(fq.observer.max_val)

    @pytest.mark.parametrize("val", [float("inf"), float("-inf"), float("nan")])
    def test_forward_non_finite_input(
        self, default_fake_quant: FakeQuantize, device, val
    ):
        """验证：输入包含 inf/nan 时的行为"""
        fq = default_fake_quant
        fq.enable_fake_quant(True)
        fq.enable_observer(False)
        with torch.no_grad():
            fq.scale.fill_(0.1)
            fq.zero_point.fill_(128.0)  # quint8

        # 获取正确的 qmin/qmax
        qmin, qmax = (
            fq.quant_min,
            fq.quant_max,
        )  # 使用 FakeQuantize 内部计算好的 qmin/qmax

        x = torch.tensor([0.0, val, 1.0], device=device)
        y = fq(x)

        assert y.shape == x.shape
        assert y.device == device
        assert torch.isfinite(y[0])
        assert torch.isfinite(y[2])

        input_val_tensor = torch.tensor(val)  # 创建 tensor 以便使用 torch.is* 函数

        if torch.isnan(input_val_tensor):
            assert torch.isnan(y[1]), "NaN input did not propagate as NaN output"
        elif torch.isinf(input_val_tensor):
            # 对于 inf/-inf，输出应被钳位到量化范围的边界并反量化
            assert torch.isfinite(
                y[1]
            ), f"{val} input did not result in finite output. Got {y[1]}"
            if val > 0:  # Positive Inf clamps to qmax
                expected_val = (qmax - fq.zero_point.item()) * fq.scale.item()
                assert torch.allclose(
                    y[1], torch.tensor(expected_val, device=device)
                ), f"+Inf input did not clamp to max value. Expected {expected_val}, Got {y[1]}"
            else:  # Negative Inf clamps to qmin
                expected_val = (qmin - fq.zero_point.item()) * fq.scale.item()
                assert torch.allclose(
                    y[1], torch.tensor(expected_val, device=device)
                ), f"-Inf input did not clamp to min value. Expected {expected_val}, Got {y[1]}"


class TestFakeQuantGradients:
    """测试 FakeQuantize 在 QAT 模式下的梯度计算 (增强)"""

    @pytest.fixture
    def fq_for_grad_test(self, default_observer_cls, device):
        """提供一个为梯度测试配置好的 FakeQuantize 实例 (在指定设备上)"""
        fq = FakeQuantize(
            observer_cls=default_observer_cls,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ).to(device)
        fq.enable_qat(QATMode.LEARNING_ONLY)  # 进入学习模式
        assert fq.scale.requires_grad is True
        assert fq.zero_point.requires_grad is True
        assert fq.scale.device == device
        assert fq.zero_point.device == device
        return fq

    # --- 辅助函数 ---
    def run_backward(
        self, fq: FakeQuantize, input_tensor: torch.Tensor, loss_scale: float = 2.0
    ):
        """辅助函数：执行前向和反向传播，使用可调损失缩放"""
        if input_tensor.grad is not None:
            input_tensor.grad.zero_()
        if fq.scale.grad is not None:
            fq.scale.grad.zero_()
        if fq.zero_point.grad is not None:
            fq.zero_point.grad.zero_()
        output = fq(input_tensor)
        loss = output.sum() * loss_scale  # 乘以 loss_scale 使梯度不总是 1
        try:
            loss.backward()
        except RuntimeError as e:
            pytest.fail(f"loss.backward() failed with error: {e}")

    # --- 梯度测试 ---
    def test_gradient_to_input(self, fq_for_grad_test: FakeQuantize, device):
        """验证：梯度是否能正确流向输入 X，即使 scale 和 zp 不需要梯度"""
        fq = fq_for_grad_test
        # 关闭 scale/zp 梯度
        fq.scale.requires_grad_(False)
        fq.zero_point.requires_grad_(False)

        # 创建需要梯度的输入
        input_tensor = torch.randn(2, 5, device=device, requires_grad=True)
        loss_scale = 2.0
        self.run_backward(fq, input_tensor, loss_scale)

        assert input_tensor.grad is not None, "梯度未流向输入 X"
        # d(output)/dX ≈ 1 (STE), d(loss)/d(output) = loss_scale
        # d(loss)/dX ≈ loss_scale * 1
        expected_grad = torch.full_like(input_tensor, loss_scale)
        assert torch.allclose(
            input_tensor.grad, expected_grad
        ), f"输入梯度值不符合预期 (应接近 {loss_scale}). Got: {input_tensor.grad}"

    def test_gradient_to_scale(self, fq_for_grad_test: FakeQuantize, device):
        """验证：梯度是否能正确流向 scale 参数"""
        fq = fq_for_grad_test
        # 关闭 input/zp 梯度
        fq.zero_point.requires_grad_(False)
        input_tensor = torch.randn(2, 5, device=device, requires_grad=False)

        self.run_backward(fq, input_tensor)

        assert fq.scale.grad is not None, "梯度未流向 scale"
        assert fq.scale.grad.device == device
        assert fq.scale.grad.abs().item() > 1e-9, "Scale 梯度接近于零 (不应为零)"
        assert input_tensor.grad is None, "Input X 不应有梯度"
        assert fq.zero_point.grad is None, "Zero Point 不应有梯度"

    def test_gradient_to_zero_point(self, fq_for_grad_test: FakeQuantize, device):
        """验证：梯度是否能正确流向 zero_point 参数"""
        fq = fq_for_grad_test
        # 关闭 input/scale 梯度
        fq.scale.requires_grad_(False)
        input_tensor = torch.randn(2, 5, device=device, requires_grad=False)

        self.run_backward(fq, input_tensor)

        assert fq.zero_point.grad is not None, "梯度未流向 zero_point"
        assert fq.zero_point.grad.device == device
        assert (
            fq.zero_point.grad.abs().item() > 1e-9
        ), "Zero Point 梯度接近于零 (不应为零)"
        assert input_tensor.grad is None, "Input X 不应有梯度"
        assert fq.scale.grad is None, "Scale 不应有梯度"

    def test_gradient_all_enabled(self, fq_for_grad_test: FakeQuantize, device):
        """验证：当 X, scale, zero_point 都需要梯度时，梯度都能正常计算"""
        fq = fq_for_grad_test
        input_tensor = torch.randn(2, 5, device=device, requires_grad=True)
        loss_scale = 2.0
        self.run_backward(fq, input_tensor, loss_scale)

        assert input_tensor.grad is not None, "梯度未流向输入 X (all enabled)"
        expected_input_grad = torch.full_like(input_tensor, loss_scale)
        assert torch.allclose(
            input_tensor.grad, expected_input_grad
        ), f"输入梯度值不符合预期 (all enabled). Got: {input_tensor.grad}"

        assert fq.scale.grad is not None, "梯度未流向 scale (all enabled)"
        assert fq.scale.grad.abs().item() > 1e-9, "Scale 梯度接近于零 (all enabled)"
        assert fq.scale.grad.device == device

        assert fq.zero_point.grad is not None, "梯度未流向 zero_point (all enabled)"
        assert (
            fq.zero_point.grad.abs().item() > 1e-9
        ), "Zero Point 梯度接近于零 (all enabled)"
        assert fq.zero_point.grad.device == device

    def test_no_gradient_when_fake_quant_disabled(
        self, fq_for_grad_test: FakeQuantize, device
    ):
        """验证：当 fake_quant 关闭时，梯度行为（STE 不生效）"""
        fq = fq_for_grad_test
        fq.enable_fake_quant(False)  # 关闭伪量化
        # 即使设为 True，也不应收到梯度
        fq.scale.requires_grad_(True)
        fq.zero_point.requires_grad_(True)

        input_tensor = torch.randn(2, 5, device=device, requires_grad=True)
        loss_scale = 2.0
        self.run_backward(fq, input_tensor, loss_scale)

        # 输入梯度应该正常（等于 d(loss)/d(output) * d(X)/dX = loss_scale * 1）
        assert input_tensor.grad is not None
        expected_input_grad = torch.full_like(input_tensor, loss_scale)
        assert torch.allclose(input_tensor.grad, expected_input_grad)
        # Scale 和 ZP 不应有梯度
        assert fq.scale.grad is None, "Scale should have no grad when fake_quant is off"
        assert (
            fq.zero_point.grad is None
        ), "ZP should have no grad when fake_quant is off"


class TestFakeQuantDeepcopy:
    def test_deepcopy_preserves_state(
        self, qat_hybrid_fake_quant: FakeQuantize, device
    ):
        """验证：deepcopy 是否能正确复制模块的状态和参数 (可能失败)"""
        fq_orig = qat_hybrid_fake_quant
        # 产生一些状态
        fq_orig.observer(torch.tensor([-0.5, 0.5], device=device))
        with torch.no_grad():
            fq_orig.scale += 0.1
            fq_orig.zero_point -= 5.0

        fq_copy = copy.deepcopy(fq_orig)
        fq_copy.to(device)  # deepcopy 后可能需要重新移动设备

        # 检查基本属性
        assert fq_copy.observer_cls == fq_orig.observer_cls
        assert fq_copy.dtype == fq_orig.dtype
        assert fq_copy.qscheme == fq_orig.qscheme
        assert fq_copy.reduce_range == fq_orig.reduce_range
        assert fq_copy.observer.reduce_range == fq_orig.observer.reduce_range
        assert fq_copy.quant_min == fq_orig.quant_min
        assert fq_copy.quant_max == fq_orig.quant_max
        assert fq_copy.eps == fq_orig.eps

        # 检查状态标志
        assert fq_copy.observer_enabled == fq_orig.observer_enabled
        assert fq_copy.fake_quant_enabled == fq_orig.fake_quant_enabled
        assert fq_copy.is_qat_learning == fq_orig.is_qat_learning
        assert fq_copy.qat_mode == fq_orig.qat_mode
        assert fq_copy.ema_alpha == fq_orig.ema_alpha

        # 检查参数值和梯度状态
        assert torch.equal(fq_copy.scale.data, fq_orig.scale.data)
        assert torch.equal(fq_copy.zero_point.data, fq_orig.zero_point.data)
        assert fq_copy.scale.requires_grad == fq_orig.scale.requires_grad
        assert fq_copy.zero_point.requires_grad == fq_orig.zero_point.requires_grad

        # 检查 observer 状态 (min_val/max_val 是 buffer)
        assert torch.equal(fq_copy.observer.min_val, fq_orig.observer.min_val)
        assert torch.equal(fq_copy.observer.max_val, fq_orig.observer.max_val)

        # 确保是不同的对象
        assert id(fq_copy) != id(fq_orig)
        assert id(fq_copy.scale) != id(fq_orig.scale)
        assert id(fq_copy.observer) != id(fq_orig.observer)
        # 如果 observer 有其他非 buffer/param 属性，也要检查是否深拷贝


class TestFakeQuantSerialization:
    """测试 FakeQuantize 的 torch.save 和 torch.load 功能"""

    def test_save_load_preserves_state(
        self, qat_hybrid_fake_quant: FakeQuantize, device
    ):
        """验证：torch.save 和 torch.load 是否能正确保存和恢复状态"""
        # 1. 获取一个配置好的 FakeQuantize 实例 (qat_hybrid_fake_quant 包含复杂状态)。
        # 2. 修改一些状态：让 observer 观察数据，手动改变 scale/zp 的值。
        # 3. 使用 torch.save 将实例保存到内存缓冲区 (io.BytesIO)。
        # 4. 使用 torch.load 从缓冲区加载回一个新的实例 fq_loaded。
        # 5. 详细比较 fq_orig 和 fq_loaded 的所有重要属性和状态，确保它们完全一致。
        #    包括：配置 (dtype, qscheme, eps, reduce_range), 状态标志 (enabled flags, qat_mode),
        #    参数值 (scale, zp), 参数梯度状态 (requires_grad), observer 内部状态 (min_val, max_val)。
        # 6. 确保加载后的实例在正确的设备上。

        fq_orig = qat_hybrid_fake_quant
        # 修改状态
        fq_orig.observer(torch.tensor([-1.2, 0.8], device=device))
        with torch.no_grad():
            fq_orig.scale += 0.01
            fq_orig.zero_point -= 2.0
        fq_orig.enable_fake_quant(False)  # 改变一个状态标志

        # 保存到内存
        buffer = io.BytesIO()
        torch.save(fq_orig, buffer)
        buffer.seek(0)  # 重置缓冲区指针到开头

        # 加载
        fq_loaded = torch.load(buffer)
        fq_loaded.to(device)  # 加载后可能需要在目标设备上

        # --- 详细比较 ---
        # 配置
        assert fq_loaded.observer_cls is fq_orig.observer_cls
        assert fq_loaded.dtype is fq_orig.dtype
        assert fq_loaded.qscheme is fq_orig.qscheme
        assert fq_loaded.reduce_range == fq_orig.reduce_range
        assert fq_loaded.eps == fq_orig.eps
        assert fq_loaded.quant_min == fq_orig.quant_min
        assert fq_loaded.quant_max == fq_orig.quant_max
        assert type(fq_loaded.observer) is type(fq_orig.observer)  # 检查 observer 类型
        assert (
            fq_loaded.observer.reduce_range == fq_orig.observer.reduce_range
        )  # 检查实例化的 observer

        # 状态标志
        assert fq_loaded.observer_enabled == fq_orig.observer_enabled
        assert (
            fq_loaded.fake_quant_enabled == fq_orig.fake_quant_enabled
        )  # 应该恢复为 False
        assert fq_loaded.is_qat_learning == fq_orig.is_qat_learning
        assert fq_loaded.qat_mode == fq_orig.qat_mode
        assert fq_loaded.ema_alpha == fq_orig.ema_alpha

        # 参数和 Buffers (比较值)
        assert torch.equal(fq_loaded.scale.data, fq_orig.scale.data)
        assert torch.equal(fq_loaded.zero_point.data, fq_orig.zero_point.data)
        assert torch.equal(fq_loaded.observer.min_val, fq_orig.observer.min_val)
        assert torch.equal(fq_loaded.observer.max_val, fq_orig.observer.max_val)

        # 参数梯度状态
        assert fq_loaded.scale.requires_grad == fq_orig.scale.requires_grad
        assert fq_loaded.zero_point.requires_grad == fq_orig.zero_point.requires_grad

        # 设备检查
        assert fq_loaded.scale.device == device
        assert fq_loaded.zero_point.device == device
        assert fq_loaded.observer.min_val.device == device
        assert fq_loaded.observer.max_val.device == device

        # 确保不是同一个对象
        assert id(fq_loaded) != id(fq_orig)
