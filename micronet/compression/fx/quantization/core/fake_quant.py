# micronet/compression/fx/quantization/core/fake_quant.py

from enum import Enum
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from .observer import MinMaxObserver
from .quant_utils import calculate_qmin_qmax


# 类型提示：一个 Observer 类或其工厂函数
ObserverCls = Callable[..., nn.Module]


class QATMode(Enum):
    LEARNING_ONLY = 1  # 仅学习
    STATS_ONLY = 2  # 仅统计
    HYBRID_EMA = 3  # 混合 EMA


class _FakeQuantizePerTensorAffineFunction(torch.autograd.Function):
    """
    **自定义 Autograd 函数：实现 Per-Tensor Affine 伪量化的核心引擎**

    **核心目的:**
    此类定义了 Per-Tensor Affine 伪量化操作的前向和反向传播行为。
    它是 `FakeQuantize` nn.Module 内部的核心计算单元。使用自定义
    `torch.autograd.Function` 的主要原因是标准 PyTorch 操作（如
    `torch.round`, `torch.clamp`）的反向传播行为不适合量化感知训练 (QAT)。
    我们需要显式地定义梯度如何流过这个模拟量化步骤。

    **关键机制:**
    1.  **伪量化 (Fake Quantization):** 在前向传播中，它模拟将浮点输入张量 (X)
        量化到较低位宽整数（由 qmin, qmax 定义），然后再反量化回浮点域。
        整个过程保持数据类型为浮点，但数值上引入了量化误差（精度损失）。
    2.  **Per-Tensor Affine:** 整个输入张量共享 *同一个* `scale`（尺度因子）和
        `zero_point`（零点）进行量化。
    3.  **直通估计器 (Straight-Through Estimator, STE):**
        *   **对于输入 X:** 在反向传播时，将输出梯度 (`grad_output`) *直接* 传递给
            输入 X 的梯度 (`grad_X`)，有效地“绕过”了 `round` 和 `clamp` 操作
            对梯度的影响（这些操作的导数要么是 0 要么是未定义的）。这使得梯度
            能够流回模型的前续层。
        *   **对于 `scale` 和 `zero_point`:** 反向传播时，根据链式法则和对量化/
            反量化过程的求导（通常是近似求导，忽略 `round` 的影响），计算
            `scale` 和 `zero_point` 的梯度。这里会考虑 `clamp` 操作的影响，
            即被钳位的值不应该对 `scale` 和 `zero_point` 的梯度产生贡献。

    **主要方法:**
    *   `forward(ctx, X, scale, zero_point, qmin, qmax, eps)`:
        执行前向伪量化计算（缩放、移位、取整、钳位、反量化）。同时，
        使用 `ctx` 对象保存反向传播所需的张量（如 X, scale, zero_point, x_q）
        和非张量参数（如 qmin, qmax, eps）。
    *   `backward(ctx, grad_output)`:
        计算并返回相对于 `forward` 方法输入的梯度。它利用 `ctx` 中保存的
        信息和传入的 `grad_output`（损失函数对伪量化输出的梯度）来计算
        `grad_X`, `grad_scale`, `grad_zero_point`。对于 `forward` 中不需
        要梯度的输入（如 qmin, qmax, eps），返回 `None`。

    **使用场景:**
    这个类通常不被用户直接调用。`FakeQuantize` nn.Module 会在 `forward`
    方法内部调用 `_FakeQuantizePerTensorAffineFunction.apply(...)` 来执行
    实际的计算和梯度关联。`FakeQuantize` 模块负责管理状态（如 QAT 模式）、
    观察者 (Observer)、参数 (scale, zero_point) 的创建和更新等。
    """

    @staticmethod
    def forward(ctx, X, scale, zero_point, qmin, qmax, eps):
        """
        定义伪量化函数的 *前向* 传播逻辑。

        Args:
            ctx: Context 对象，用于存储信息以便在反向传播中使用。
                 PyTorch 自动处理此参数。
            X (torch.Tensor): 输入的浮点张量。
            scale (torch.Tensor): 量化尺度因子 (通常是标量)。
            zero_point (torch.Tensor): 量化零点 (通常是标量，类型可能为浮点或整数，但在此计算中视为浮点)。
            qmin (int): 量化范围的最小值 (例如，0 for quint8, -128 for qint8)。
            qmax (int): 量化范围的最大值 (例如，255 for quint8, 127 for qint8)。
            eps (float): 一个很小的正数，用于防止在 scale 接近零时发生除以零的错误。

        Returns:
            torch.Tensor: 伪量化后的输出张量 (X_dq)，其数值上模拟了量化和反量化的效果，
                          但仍然是浮点类型，并且通过 ctx 关联了反向传播逻辑。
        """

        # --- 1. 缩放与移位 (Scale and Shift) ---
        # 将输入张量 X 映射到量化空间。
        # 首先，通过除以 (scale + eps) 来缩放输入。添加 eps 是为了数值稳定性，
        # 防止当 scale 非常接近 0 时出现除以零的情况。
        # 然后，加上 zero_point 将其移位。
        # 结果 scaled_shifted_X 是一个浮点张量，其值大致分布在量化整数范围内。
        scaled_shifted_X = X / (scale + eps) + zero_point

        # --- 2. 四舍五入 (Round) ---
        # 将经过缩放和移位的浮点值四舍五入到最接近的整数。
        # 这是模拟量化到离散整数值的关键步骤。
        # 注意：torch.round() 操作本身是不可导的（或者说梯度几乎处处为零），
        # 这就是为什么需要自定义 backward 函数并使用直通估计器 (STE)。
        rounded_X = torch.round(scaled_shifted_X)

        # --- 3. 钳位/截断 (Clamp) ---
        # 将四舍五入后的整数值限制在有效的量化范围 [qmin, qmax] 之内。
        # 这模拟了定点数表示中可能发生的饱和（saturation）现象。
        # 超出范围的值会被截断到边界值 qmin 或 qmax。
        # x_q 现在包含了模拟的量化整数值。
        x_q = torch.clamp(rounded_X, qmin, qmax)

        # --- 4. 反量化 (Dequantize) ---
        # 将钳位后的整数值 x_q 转换回浮点域，得到伪量化张量 X_dq。
        # 首先，减去零点 zero_point。
        # 然后，乘以尺度因子 scale。
        # X_dq 是原始输入 X 的一个近似，其数值精度受到了模拟量化过程的影响。
        X_dq = (x_q - zero_point) * scale

        # --- 5. 保存信息以备反向传播使用 (Save for Backward) ---
        # 使用 ctx.save_for_backward() 保存需要在 backward 方法中使用的 *张量*。
        # 保存 X：用于 STE 计算 grad_X。
        # 保存 scale 和 zero_point：用于计算它们各自的梯度 grad_scale 和 grad_zero_point。
        # 保存 x_q (钳位后的量化整数值): 用于在 backward 中计算梯度掩码 (mask)，
        #                          以确定哪些元素因钳位而不应贡献梯度给 scale 和 zero_point。
        # 注意：保存的张量将在 backward 方法中通过 ctx.saved_tensors 访问。
        ctx.save_for_backward(
            X, scale, zero_point, rounded_X, x_q
        )  # 注意: 保存的是钳位后的 x_q

        # 将非张量参数直接存储为 ctx 的属性，以便在 backward 中访问。
        ctx.qmin = qmin
        ctx.qmax = qmax
        ctx.eps = eps

        # --- 6. 返回伪量化结果 ---
        # 返回最终的伪量化张量 X_dq。
        # 这个张量将流向模型的下一层，并在反向传播时触发调用我们定义的 backward 方法。
        return X_dq

    @staticmethod
    def backward(ctx, grad_output):
        """
        定义伪量化函数的 *反向* 传播逻辑。

        Args:
            ctx: Context 对象，用于存储和检索前向传播中保存的张量和参数。
            grad_output (torch.Tensor): 从计算图后续层传回的梯度，形状与前向传播的输出相同。

        Returns:
            Tuple[Optional[torch.Tensor], ...]: 对应于前向传播 `apply` 方法 *所有* 输入参数的梯度。
                                                如果某个输入不需要梯度，则返回 None。
                                                顺序必须严格匹配 `forward` 方法的输入参数顺序：
                                                (grad_X, grad_scale, grad_zero_point, grad_qmin, grad_qmax, grad_eps, grad_dtype, grad_device)
                                                其中 qmin, qmax, eps, dtype, device 通常不需要梯度，返回 None。
        """
        # --- 1. 恢复前向传播保存的张量和参数 ---
        # 从 ctx 对象中解包在前向传播时使用 save_for_backward 保存的张量。
        # 这些张量对于计算反向梯度至关重要。
        # X: 原始输入张量
        # scale: 使用的量化尺度因子
        # zero_point: 使用的量化零点
        # x_q: 量化后的（未钳位的）整数值张量 (注意：这里保存的是未钳位的值，用于计算梯度掩码)
        X, scale, zero_point, rounded_X, x_q = (
            ctx.saved_tensors
        )  # 注意：这里假设保存的是 x_q_unc (未钳位的)

        # 从 ctx 对象中恢复非张量参数 (在前向传播时设置到 ctx 上的属性)。
        qmin, qmax = ctx.qmin, ctx.qmax  # 量化范围的最小值和最大值
        eps = ctx.eps  # 用于防止除以零的小值

        # --- 2. 检查哪些输入需要梯度 ---
        # ctx.needs_input_grad 是一个布尔元组，对应 forward 方法的每个输入。
        # 它告诉我们对于哪些输入需要计算梯度 (即其 requires_grad=True)。
        # 这有助于优化计算，避免为不需要梯度的输入计算梯度。
        # 顺序对应 forward 的输入: (X, scale, zero_point, qmin, qmax, eps, dtype, device)
        needs_grad_X, needs_grad_scale, needs_grad_zp, *_ = ctx.needs_input_grad

        # --- 3. 初始化梯度变量 ---
        # 将所有可能的梯度输出初始化为 None。
        # 稍后，只有当对应的输入需要梯度时，才会计算并赋值。
        grad_X = grad_scale = grad_zero_point = None

        # --- 4. 计算输入 X 的梯度 (grad_X) ---
        # 应用直通估计器 (STE) 的反向传播部分。
        # STE 的核心思想是：在前向传播中使用非线性、不可导的操作 (如 round, clamp)，
        # 但在反向传播时，将该操作视为恒等函数 (identity function)。
        # 因此，d(Loss)/d(X) ≈ d(Loss)/d(Output) * d(Output)/d(X)
        # 由于我们将伪量化视为恒等函数 (对于输入 X 的梯度)，d(Output)/d(X) ≈ 1。
        # 所以，输入 X 的梯度就近似等于输出传回的梯度 grad_output。
        if needs_grad_X:
            # 直接将上游传回的梯度 grad_output 赋给 grad_X。
            grad_X = grad_output

        # --- 5. 计算 scale 和 zero_point 的梯度 ---
        # 仅当 scale 或 zero_point 需要梯度时才执行这部分计算 (通常在 QAT 模式下)。
        if needs_grad_scale or needs_grad_zp:
            # --- 5.1 创建梯度掩码 (Mask) ---
            X_q = rounded_X
            X_q_clamped = x_q
            # 梯度只应流经那些 *未被钳位* 的元素。
            # 如果一个元素在前向传播时被钳位到了 qmin 或 qmax，
            # 那么 scale 或 zero_point 的微小变动不会影响最终的钳位结果 X_q_clamped，
            # 因此这些被钳位元素的梯度贡献应为 0。
            # mask 为 True 的位置表示该元素 *在量化范围之内* (未被钳位)。
            mask = (X_q >= qmin) & (X_q <= qmax)

            # --- 5.2 计算 grad_scale ---
            # d(Loss)/d(scale) ≈ d(Loss)/d(X_dq) * d(X_dq)/d(scale)
            # 其中 d(Loss)/d(X_dq) ≈ grad_output (因为 STE 将 X_dq 到 Output 近似为恒等)
            # 而 d(X_dq)/d(scale) = d((X_q_clamped - zero_point) * scale) / d(scale)
            #                     ≈ X_q_clamped - zero_point (忽略 round 对 scale 的复杂导数，这是 QAT 的关键近似)
            # 结合 mask，只考虑未被钳位的元素。
            if needs_grad_scale:
                # grad_output * (X_q_clamped - zero_point) 计算每个元素的梯度贡献。
                # * mask 将被钳位元素的贡献置零。
                # .sum() 将所有元素的贡献累加起来，因为 scale 是一个标量参数。
                grad_scale = (grad_output * (X_q_clamped - zero_point) * mask).sum()

            # --- 5.3 计算 grad_zero_point ---
            # d(Loss)/d(zero_point) ≈ d(Loss)/d(X_dq) * d(X_dq)/d(zero_point)
            # 其中 d(Loss)/d(X_dq) ≈ grad_output
            # 而 d(X_dq)/d(zero_point) = d((X_q_clamped - zero_point) * scale) / d(zero_point)
            #                          ≈ -scale (同样忽略 round 对 zero_point 的影响，这是 QAT 的近似)
            # 结合 mask。
            if needs_grad_zp:
                # grad_output * (-scale) 计算每个元素的梯度贡献 (注意负号)。
                # * mask 将被钳位元素的贡献置零。
                # .sum() 将所有元素的贡献累加起来，因为 zero_point 通常也是标量 (per-tensor)。
                # 注意：这里假定了 zero_point 是一个浮点数或在计算中表现得像浮点数。
                grad_zero_point = (grad_output * (-scale) * mask).sum()

        # --- 6. 返回梯度 ---
        # 返回的梯度元组必须严格按照 forward 方法输入的顺序。
        # 对于不需要梯度的输入 (或未计算的梯度)，返回 None。
        # (grad_X, grad_scale, grad_zero_point, grad_qmin, grad_qmax, grad_eps, grad_dtype, grad_device)
        return grad_X, grad_scale, grad_zero_point, None, None, None, None, None


class FakeQuantize(nn.Module):
    """
    **核心伪量化 (Fake Quantization) 模块**

    **概述:**
    该模块是量化框架的关键组件，用于在模型训练（QAT）或评估（PTQ）过程中
    *模拟* 量化操作对模型激活或权重的影响。它并 *不* 真正将数据类型转换为
    低精度整数（如 int8），而是保持数据为浮点类型，但其数值被调整以匹配
    量化后再反量化 (quantize-dequantize) 的效果，从而引入量化误差。

    **主要用途:**
    1.  **量化感知训练 (Quantization-Aware Training, QAT):** 在训练过程中插入此
        模块，模型可以学习适应量化引入的噪声和精度损失，从而在最终转换为
        真实量化模型（如 int8）时获得更好的性能。梯度通过此模块使用
        直通估计器 (Straight-Through Estimator, STE) 技术进行反向传播。
    2.  **后训练量化 (Post-Training Quantization, PTQ):**
        *   **校准 (Calibration):** 在 PTQ 的校准阶段，启用内部的 `Observer` 来
            收集输入数据的统计信息（例如最小值、最大值），以确定合适的量化
            参数（`scale` 和 `zero_point`）。此时通常禁用伪量化本身。
        *   **评估 (Evaluation):** 在获得量化参数后，可以启用伪量化（禁用
            Observer）来评估模拟量化对模型准确率的影响，而无需实际转换模型。

    **核心机制:**
    *   **内部观察者 (Observer):** 包含一个可配置的 Observer 实例（默认为
        `MinMaxObserver`），用于在需要时收集输入数据的统计信息。
    *   **量化参数 (Scale & Zero-Point):** 存储 `scale`（尺度因子）和
        `zero_point`（零点）作为 `nn.Parameter`。这些参数决定了如何将浮点值
        映射到量化范围。
        *   在 PTQ 中，这些参数通常根据 Observer 收集的统计信息计算得出且固定。
        *   在 QAT 中，这些参数可以被设置为可学习的，通过反向传播进行优化，
            也可以结合统计信息进行更新（取决于 QAT 模式）。
    *   **可配置模式:** 通过 `enable_*` 方法可以灵活控制模块的行为：
        *   `observer_enabled`: 是否运行 Observer 收集统计信息。
        *   `fake_quant_enabled`: 是否执行伪量化操作（模拟量化）。
        *   `is_qat_learning`: 是否处于 QAT 阶段。
        *   `qat_mode`: QAT 的具体模式（仅学习参数、仅依赖统计、混合）。
    *   **自定义 Autograd 函数:** 内部使用
        `_FakeQuantizePerTensorAffineFunction` 来执行实际的伪量化计算。这个
        自定义函数确保了：
        *   前向传播执行正确的伪量化步骤（缩放、移位、取整、钳位、反量化）。
        *   反向传播时，梯度能够通过 STE 正确流向输入 `X`，并且根据（近似）
            导数计算 `scale` 和 `zero_point` 的梯度，支持 QAT。

    **支持的量化方案:**
    *   目前主要针对 **Per-Tensor Affine** 量化方案实现，即整个张量共享一对
        `scale` 和 `zero_point`。

    **配置与使用:**
    *   通常不由用户直接实例化，而是通过量化框架的 `QConfig`（量化配置）
        指定，并由 `Quantizer`（或其他模型准备工具）自动插入到模型的适当位置。
    *   可以通过 `enable_*` 方法在不同阶段（校准、训练、评估）切换其行为模式。

    **关键属性 (初始化时确定或可配置):**
    *   `observer`: 内部使用的 Observer 实例。
    *   `dtype`: 目标量化数据类型 (e.g., `torch.quint8`, `torch.qint8`)。
    *   `qscheme`: 目标量化方案 (e.g., `torch.per_tensor_affine`)。
    *   `quant_min`, `quant_max`: 根据 `dtype` 和 `reduce_range` 计算出的
        目标量化范围的整数边界。
    *   `scale`: 量化尺度因子 (`nn.Parameter`)。
    *   `zero_point`: 量化零点 (`nn.Parameter`, 存储为 float 但概念上是整数)。
    *   `eps`: 用于防止除以零的小常数。
    *   `observer_enabled` (bool): 控制 Observer 是否激活。
    *   `fake_quant_enabled` (bool): 控制伪量化是否执行。
    *   `is_qat_learning` (bool): 标记是否处于 QAT 训练阶段。
    *   `qat_mode` (Optional[QATMode]): QAT 的具体模式。
    *   `ema_alpha` (Optional[float]): HYBRID_EMA 模式下的指数移动平均系数。

    **示例状态 (通过 `extra_repr` 查看):**
    *   `[OBSERVING (PTQ Calib?)]`: PTQ 校准阶段，收集统计信息。
    *   `[FAKE_QUANT (PTQ Infer?)]`: PTQ 评估阶段，使用固定参数模拟量化。
    *   `[QAT_MODE=LEARNING_ONLY]`: QAT 阶段，仅通过梯度学习参数。
    *   `[QAT_MODE=STATS_ONLY]`: QAT 阶段，参数根据统计信息计算。
    *   `[QAT_MODE=HYBRID_EMA, EMA_ALPHA=...]`: QAT 阶段，混合模式。
    *   `[INACTIVE]`: 模块不起作用，相当于 Identity。
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
        "is_qat_learning",  # 表示是否处于 QAT 阶段 (学习或统计)
        "qat_mode",  # QAT 期间的具体模式
        "ema_alpha",  # EMA 更新因子 (如果使用 HYBRID_EMA)
    ]

    def __init__(
        self,
        observer_cls: ObserverCls = MinMaxObserver,
        dtype: torch.dtype = torch.quint8,
        qscheme: torch.qscheme = torch.per_tensor_affine,
        reduce_range: bool = False,
        initial_scale: float = 1.0,
        initial_zero_point: int = 0,
        eps: float = torch.finfo(torch.float32).eps,
    ):
        """
        初始化 FakeQuantize 模块。

        Args:
            observer_cls (Type[nn.Module]): 用于创建内部 Observer 实例的类。
                默认为 MinMaxObserver。Observer 需要实现 `forward` 方法来接收
                输入并更新统计信息，并且需要有 `calculate_qparams` 方法。
            dtype (torch.dtype): 目标量化的数据类型，如 torch.quint8（无符号 8 位整型）
                或 torch.qint8（有符号 8 位整型）。用于确定量化范围 [qmin, qmax]。
            qscheme (torch.qscheme): 量化方案。目前主要支持对称
                (torch.per_tensor_symmetric) 和非对称 (torch.per_tensor_affine)。
                注意：当前实现主要基于 Affine，Symmetric 可能需要调整 zero_point 处理。
            reduce_range (bool): 是否缩减量化范围。通常用于权重，例如将
                int8 的 [-128, 127] 缩减到 [-127, 127]。
            initial_scale (float): scale 参数的初始值。
            initial_zero_point (int): zero_point 参数的初始值。
            eps (float): 一个小的 epsilon 值，用于在计算中防止除以零，
                特别是在计算 `X / (scale + eps)` 时。
        """
        super().__init__()

        # 实例化内部 Observer
        self.observer = observer_cls(
            dtype=dtype, qscheme=qscheme, reduce_range=reduce_range, eps=eps
        )

        # 保存核心配置作为实例属性，以便序列化和 deepcopy
        self.dtype = dtype
        self.qscheme = qscheme
        self.eps = eps

        # 获取量化范围 [qmin, qmax]
        # calculate_qmin_qmax 是一个辅助函数，根据 dtype 和 reduce_range 返回整数范围
        self.quant_min, self.quant_max = calculate_qmin_qmax(self.dtype, reduce_range)

        # 初始化 scale 和 zero_point 为可学习参数 (nn.Parameter)
        self.scale = Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        # Zero-point 概念上是整数，但为了方便梯度计算和 EMA 更新，通常存储为 float Parameter
        self.zero_point = Parameter(
            torch.tensor(initial_zero_point, dtype=torch.float32)
        )

        # 内部状态标志
        # 控制 Observer 是否收集统计信息
        self.observer_enabled: bool = True
        # 控制是否执行伪量化操作 (模拟量化效果)
        self.fake_quant_enabled: bool = False
        # 标记是否处于 QAT 训练阶段
        self.is_qat_learning: bool = False
        # QAT 的具体模式 (影响参数如何更新)
        self.qat_mode: Optional[QATMode] = None
        # HYBRID_EMA 模式下的 EMA alpha 值
        self.ema_alpha: Optional[float] = None

        # 根据初始状态（非 QAT）设置 scale 和 zero_point 的 requires_grad
        self._update_param_learning_state()

    # --- 状态更新和控制方法 ---
    def _update_param_learning_state(self):
        """根据 QAT 状态和模式更新 scale/zp 的 requires_grad"""
        # 参数是否需要梯度？当处于 QAT 且模式允许学习时 (LEARNING_ONLY 或 HYBRID_EMA)
        needs_grad = self.is_qat_learning and (
            self.qat_mode == QATMode.LEARNING_ONLY
            or self.qat_mode == QATMode.HYBRID_EMA
        )
        # 使用 requires_grad_() 进行原地修改
        self.scale.requires_grad_(needs_grad)
        self.zero_point.requires_grad_(needs_grad)

    def enable_observer(self, enabled: bool = True) -> "FakeQuantize":
        """启用或禁用内部 Observer。返回 self 以支持链式调用。"""
        self.observer_enabled = enabled
        # 如果禁用了 Observer 但 QAT 模式依赖它，则发出警告或错误
        if not enabled and self.is_qat_learning:
            if (
                self.qat_mode == QATMode.STATS_ONLY
                or self.qat_mode == QATMode.HYBRID_EMA
            ):
                # 在 QAT 且需要统计信息的模式下禁用 Observer 是无效的
                raise ValueError(
                    f"禁用 Observer 与 {self.qat_mode.name} QAT 模式不兼容"
                )
        return self

    def enable_fake_quant(self, enabled: bool = True) -> "FakeQuantize":
        """启用或禁用伪量化计算。返回 self 以支持链式调用。"""
        self.fake_quant_enabled = enabled
        return self

    def enable_qat(
        self,
        enabled: bool = True,
        qat_mode: QATMode = QATMode.LEARNING_ONLY,
        inherit_qparams: bool = True,
        ema_alpha: Optional[float] = 0.99,
    ) -> "FakeQuantize":
        """
        启用或禁用量化感知训练 (QAT) 模式。

        Args:
            enabled (bool): 是否启用 QAT。
            qat_mode (QATMode): QAT 的具体模式 (LEARNING_ONLY, STATS_ONLY, HYBRID_EMA)。
            inherit_qparams (bool): 进入 QAT 时，是否尝试从 Observer 计算的
                qparams 初始化 scale 和 zero_point。
            ema_alpha (Optional[float]): 如果 `qat_mode` 是 HYBRID_EMA，
                则使用此 alpha 值进行指数移动平均更新。

        Returns:
            FakeQuantize: 返回 self 以支持链式调用。

        Raises:
            ValueError: 如果 HYBRID_EMA 模式下提供了无效的 ema_alpha。
        """
        self.is_qat_learning = enabled
        self.qat_mode = qat_mode if enabled else None
        # 仅在启用 QAT 且模式为 HYBRID_EMA 时存储 ema_alpha
        self.ema_alpha = (
            ema_alpha if enabled and qat_mode == QATMode.HYBRID_EMA else None
        )

        if enabled:
            # 进入 QAT 时，通常总是启用伪量化
            self.fake_quant_enabled = True

            # 根据 QAT 模式调整 Observer 状态和检查 ema_alpha
            if qat_mode == QATMode.STATS_ONLY or qat_mode == QATMode.HYBRID_EMA:
                # 这两种模式依赖 Observer 收集统计信息
                self.observer_enabled = True
                if not (0.0 <= ema_alpha <= 1.0) and qat_mode == QATMode.HYBRID_EMA:
                    raise ValueError(
                        "HYBRID_EMA 模式需要有效的 ema_alpha (0 <= alpha <= 1)"
                    )
            else:  # LEARNING_ONLY 模式
                # 通常不依赖 Observer，可以禁用它以节省计算，除非用户之前就禁用了它
                if not self.observer_enabled:
                    pass  # 如果用户之前手动禁用了，保持禁用
                else:
                    self.observer_enabled = False  # 否则，在 LEARNING_ONLY 模式下禁用

            # 尝试从 Observer 的当前状态继承量化参数
            if inherit_qparams:
                try:
                    # 检查 observer 是否有有效的 min/max 值
                    if torch.isfinite(self.observer.min_val) and torch.isfinite(
                        self.observer.max_val
                    ):
                        # 调用 observer 的方法计算 qparams
                        scale, zp = self.observer.calculate_qparams()
                        # 使用 no_grad() 避免影响梯度，并用 copy_ 更新参数值
                        with torch.no_grad():
                            self.scale.copy_(scale)
                            # zero_point 参数是 float，Observer 返回的可能是 int，需要转换
                            self.zero_point.copy_(zp.to(torch.float32))
                    else:
                        # Observer 还没有收集到有效数据，无法继承，保持初始值
                        print(
                            "Warning: Observer has no finite min/max, cannot inherit qparams."
                        )
                except Exception as e:
                    # 如果 calculate_qparams 失败或 Observer 状态无效，打印警告并跳过
                    # print(f"Warning: Failed to inherit qparams from observer: {e}")
                    pass

        else:  # 禁用 QAT
            # 退出 QAT 时，通常也禁用 Observer
            self.observer_enabled = False

        self._update_param_learning_state()
        return self

    # --- 量化参数计算与更新 ---
    @torch.jit.export  # 标记为可 TorchScript 导出的方法
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        强制从当前 Observer 的统计信息计算量化参数，并更新模块的 scale 和 zero_point。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 计算得到的 scale 和 zero_point。
        """
        # 调用 observer 的核心方法来计算参数
        scale, zp = self.observer.calculate_qparams()
        # 在 no_grad 上下文中更新模块的参数
        with torch.no_grad():
            self.scale.copy_(scale)
            self.zero_point.copy_(zp.to(torch.float32))
        return self.scale, self.zero_point

    def _apply_ema_update(self):
        """
        (内部方法) 如果处于 HYBRID_EMA 模式，则应用指数移动平均更新
        scale 和 zero_point。
        """
        # 仅在 HYBRID_EMA 模式且 ema_alpha 有效时执行
        if self.qat_mode != QATMode.HYBRID_EMA or self.ema_alpha is None:
            return

        try:
            # 从 observer 获取基于当前统计信息的 qparams
            scale_stat, zp_stat = self.observer.calculate_qparams()

            # 仅当统计信息有效时才进行更新
            if torch.isfinite(self.observer.min_val) and torch.isfinite(
                self.observer.max_val
            ):
                with torch.no_grad():
                    alpha = self.ema_alpha
                    # EMA 更新公式: new_val = alpha * old_val + (1 - alpha) * stat_val
                    self.scale.copy_(alpha * self.scale.data + (1 - alpha) * scale_stat)
                    self.zero_point.copy_(
                        alpha * self.zero_point.data
                        + (1 - alpha) * zp_stat.to(torch.float32)
                    )
        except Exception as e:
            # 如果计算失败（例如 observer 状态无效），打印警告并跳过更新
            # print(f"Warning: EMA update failed: {e}")
            pass

    # --- 核心前向传播 ---
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        执行模块的核心前向传播逻辑。

        根据模块的当前状态 (observer_enabled, fake_quant_enabled, is_qat_learning, qat_mode)，
        执行以下操作：
        1.  如果 `observer_enabled`，则将输入 `X` 传递给内部 `Observer` 以收集统计信息。
        2.  如果处于 QAT 模式 (`is_qat_learning`)：
            *   如果是 `STATS_ONLY` 模式，调用 `calculate_qparams` 使用最新统计更新 scale/zp。
            *   如果是 `HYBRID_EMA` 模式，调用 `_apply_ema_update` 进行 EMA 更新。
            *   `LEARNING_ONLY` 模式下，参数主要通过反向传播学习，此处不主动更新。
        3.  如果 `fake_quant_enabled`，则执行伪量化操作：
            *   使用当前的 `scale`, `zero_point`, `quant_min`, `quant_max` 和 `eps`
              调用内部的 `_FakeQuantizePerTensorAffineFunction.apply()`。
            *   此函数执行伪量化计算并处理梯度（STE）。
            *   返回伪量化后的张量 `X_dq`。
        4.  如果 `fake_quant_enabled` 为 `False`，则直接返回原始输入 `X` (模块相当于 Identity)。

        Args:
            X (torch.Tensor): 输入的浮点张量。

        Returns:
            torch.Tensor: 经过（可能发生的）观察、参数更新和伪量化处理后的输出张量。
                          数据类型仍为浮点。
        """
        # --- 1. Observer 运行 (如果启用) ---
        if self.observer_enabled:
            self.observer(X)  # 将输入传递给 observer 以更新其内部统计信息

        # --- 2. QAT 模式下的参数更新 (如果启用) ---
        if self.is_qat_learning:
            if self.qat_mode == QATMode.STATS_ONLY:
                # 强制使用最新统计信息更新参数 (参数本身不可学习)
                self.calculate_qparams()
            elif self.qat_mode == QATMode.HYBRID_EMA:
                # 应用 EMA 更新 (参数也可学习)
                self._apply_ema_update()
            # LEARNING_ONLY 模式下，参数通过梯度学习，此处不操作

        # --- 3. 伪量化运行 (如果启用) ---
        if self.fake_quant_enabled:
            # 调用自定义 Autograd 函数执行伪量化
            # 将核心计算委托给 _FakeQuantizePerTensorAffineFunction
            # .apply() 方法会自动处理前向计算和反向传播梯度
            X_dq = _FakeQuantizePerTensorAffineFunction.apply(
                X,  # 输入张量
                self.scale,  # 当前 scale (Parameter)
                self.zero_point,  # 当前 zero_point (Parameter)
                float(self.quant_min),  # 量化下界 (转为 float 以匹配 Function 签名预期)
                float(self.quant_max),  # 量化上界 (转为 float)
                self.eps,  # Epsilon 值
            )
            return X_dq  # 返回伪量化后的结果
        else:
            # 如果伪量化被禁用，模块行为类似 Identity
            return X

    # --- 其他辅助方法 ---
    def extra_repr(self) -> str:
        """提供模块状态的额外字符串表示，用于打印模块信息。"""
        obs_repr = repr(self.observer)
        # 构建状态字符串
        state = []
        if self.is_qat_learning:
            state.append(f"QAT_MODE={self.qat_mode.name}")
            if self.qat_mode == QATMode.HYBRID_EMA:
                state.append(f"EMA_ALPHA={self.ema_alpha:.3f}")
        elif self.observer_enabled:
            state.append("OBSERVING")  # 可能在 PTQ 校准
        elif self.fake_quant_enabled:
            state.append("FAKE_QUANT")  # 可能在 PTQ 评估
        else:
            state.append("INACTIVE")  # 模块当前不起作用

        # 指示参数是否可学习
        param_state = "(Learnable)" if self.scale.requires_grad else "(Fixed)"
        # 格式化最终输出
        return (
            f"state=[{', '.join(state)}], "
            f"scale={self.scale.item():.4f}{param_state}, "
            f"zero_point={self.zero_point.item():.4f}{param_state}, "
            f"observer=({obs_repr})"
        )

    def reset_observer_stats(self):
        """方便地重置内部 observer 的统计信息 (如果 observer 支持)。"""
        # 检查 observer 是否有 reset_stats 方法
        if hasattr(self.observer, "reset_stats"):
            self.observer.reset_stats()

    def __deepcopy__(self, memo):
        # 创建一个新的实例，具有相同的初始化参数
        new_instance = FakeQuantize(
            observer_cls=type(self.observer),
            dtype=self.dtype,
            qscheme=self.qscheme,
            reduce_range=getattr(self.observer, "reduce_range", False),
            initial_scale=self.scale.item(),
            initial_zero_point=int(round(self.zero_point.item())),
            eps=self.eps,
        )
        memo[id(self)] = new_instance  # 存入 memo 防止循环引用

        # 复制内部状态
        new_instance.observer_enabled = self.observer_enabled
        new_instance.fake_quant_enabled = self.fake_quant_enabled
        new_instance.is_qat_learning = self.is_qat_learning
        new_instance.qat_mode = self.qat_mode
        new_instance.ema_alpha = self.ema_alpha

        # 复制 observer 的状态 (如果 observer 有状态需要复制)
        if hasattr(self.observer, "state_dict") and hasattr(
            new_instance.observer, "load_state_dict"
        ):
            new_instance.observer.load_state_dict(self.observer.state_dict())
        elif hasattr(self.observer, "min_val") and hasattr(
            new_instance.observer, "min_val"
        ):  # 对 MinMaxObserver 特殊处理
            new_instance.observer.min_val.copy_(self.observer.min_val)
            new_instance.observer.max_val.copy_(self.observer.max_val)

        # 复制 Parameter 的状态 (值和 requires_grad)
        new_instance.scale.data.copy_(self.scale.data)
        new_instance.zero_point.data.copy_(self.zero_point.data)
        new_instance._update_param_learning_state()  # 确保 requires_grad 状态正确

        return new_instance
