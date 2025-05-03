# micronet/compression/fx/quantization/core/quantizer.py

import copy
import logging
import sys

import torch.nn as nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule

from .qconfig import QConfig

from .fake_quant import FakeQuantize
from .graph_utils import (
    fuse_conv_linear_bn_fx,
    is_quantizable_weight_module,
    is_quantizable_activation_module,
    is_quantizable_activation_function,
    is_quantizable_activation_method,
    _colorize,
    COLOR_DEBUG,
    COLOR_INFO,
    COLOR_WARN,
    COLOR_ERROR,
    COLOR_SUCCESS,
    COLOR_MODULE,
    COLOR_OPERATOR,
    COLOR_NODE,
    COLOR_BOLD,
    COLOR_CYAN,
    COLOR_PHASE,
    COLOR_ACTION,
    COLOR_REASON,
    COLOR_TARGET,
    COLOR_INPUT,
    graph_utils_logger,  # 导入 graph_utils 的日志记录器以便统一控制级别
)

# --- 配置此模块的日志记录器 ---
logger = logging.getLogger("micronet.fx.quantizer")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


class Quantizer:
    """
    **核心功能:**
    自动化地将一个标准的 PyTorch `nn.Module` 模型转换为适合进行量化（PTQ 或 QAT）的
    `torch.fx.GraphModule` 结构。

    **工作目标:**
    此类旨在解决手动修改模型以适应量化流程的复杂性和易错性。它通过分析模型的计算图，
    并根据用户提供的量化配置 (`QConfig`)，在图的关键位置自动插入“伪量化”节点
    (`FakeQuantize`)。这些节点在后续的量化流程中起着至关重要的作用。

    **主要职责与处理流程 (通过 `prepare` 方法实现):**

    1.  **模型表示转换 (符号追踪):**
        *   使用 `torch.fx.symbolic_trace` 将输入的 PyTorch 模型（Python 代码）转换为
            一个显式的、可操作的计算图表示 (FX Graph)。这是所有后续操作的基础。
        *   **注意:** 此过程通常要求模型是 "FX-traceable" 的，即不包含过于复杂的 Python
            控制流（虽然对简单条件和循环可能有一定支持，但复杂情况可能失败）。追踪
            总是在 `eval()` 模式下进行。

    2.  **模型结构优化 (可选的 BN 融合):**
        *   如果 `fuse_bn=True` (默认行为依赖于初始化设置)，则会自动扫描追踪后的图，
            查找常见的、可融合的模式，主要是卷积层 (Conv1d/2d/3d) 或线性层 (Linear)
            紧跟着批归一化层 (BatchNorm1d/2d/3d)。
        *   识别出的模式会被融合：将 BN 层的计算（缩放和平移）合并到前面的
            Conv/Linear 层的权重和偏置中。这可以减少计算量并提高量化后的精度。
        *   融合操作会直接修改 FX Graph 和关联的模块实例。

    3.  **量化节点插入 (核心逻辑):**
        *   遍历（可能已融合的）FX Graph 中的每个节点。
        *   **权重 (`weight`) 量化:**
            *   识别出定义在 `graph_utils.is_quantizable_weight_module` 中的可量化层
              (如 `nn.Conv2d`, `nn.Linear`)。
            *   如果用户在 `QConfig` 中提供了 `weight` 的 `FakeQuantize` 工厂函数，
              则会为该层的 `weight` 属性创建一个 `FakeQuantize` 实例。
            *   在图中插入对这个 `FakeQuantize` 模块的调用，使其作用于原始权重。
            *   修改原始层节点的输入，使其使用伪量化后的权重。
        *   **激活 (`activation`) 量化:**
            *   识别出定义在 `graph_utils.is_quantizable_activation_*` 中的层、函数
              或方法 (如 `nn.ReLU`, `torch.add`, `tensor.relu`) 的输出，以及模型的
              输入占位符 (`placeholder`)。
            *   如果用户在 `QConfig` 中提供了 `activation` 的 `FakeQuantize` 工厂函数，
              则会为这些节点的输出创建一个 `FakeQuantize` 实例。
            *   在图中，在这些产生需要量化的激活值的节点 *之后* 插入对 `FakeQuantize`
              模块的调用。
            *   修改图中所有原来使用该激活值的下游节点，使其转而使用 `FakeQuantize`
              模块的输出。
        *   **`FakeQuantize` 的作用:**
            *   这些插入的 `FakeQuantize` 模块本身是 `nn.Module`。
            *   它们的核心功能是 *模拟* 量化操作（量化 -> 钳位 -> 反量化），但输入
              输出仍然是浮点数。
            *   它们内部通常包含一个 `Observer`，用于在 PTQ 校准阶段收集数据的
              统计信息 (min/max)。
            *   它们也包含 `scale` 和 `zero_point` 参数。在 QAT 阶段，这些参数可以
              被设置为可学习的 (`nn.Parameter`)，并通过反向传播进行优化。
            *   在 QAT 中，它们使用直通估计器 (STE) 来允许梯度流过不可导的量化操作。

    4.  **图的最终处理与返回:**
        *   在所有修改完成后，对图进行验证 (`lint`) 确保其结构有效。
        *   将最终的 FX Graph 和更新后的模块字典重新包装成一个新的
            `torch.fx.GraphModule` 实例。
        *   返回这个准备好的 `GraphModule`。**重要的是，原始输入模型不会被修改。**

    **用户需要提供的关键信息 (`QConfig` 或 `QConfigMapping`):**
    `QConfig` 对象 (或更灵活的 `QConfigMapping`) 是指导 `Quantizer` 如何插入
    `FakeQuantize` 节点的蓝图。它至少需要指定：
    *   `activation`: 一个工厂函数 (或类)，当被调用时，返回一个用于激活量化的
                     `FakeQuantize` 实例 (或者 `None` 表示不量化激活)。
    *   `weight`: 一个工厂函数 (或类)，当被调用时，返回一个用于权重量化的
                  `FakeQuantize` 实例 (或者 `None` 表示不量化权重)。
    `QConfigMapping` 允许对不同类型的模块或特定名称的模块指定不同的量化配置，提供
    比单一 `QConfig` 更精细的控制。

    **关键属性 (Attributes):**
    *   `qconfig` (`Union[QConfig, QConfigMapping]`):
        存储传递给构造函数的量化配置。这决定了哪些模块/操作会被量化以及如何量化
        (即使用哪个 `FakeQuantize` 工厂)。
    *   `debug` (`bool`):
        控制是否启用详细的调试日志输出。如果为 `True`，则在控制台打印追踪、融合
        和量化节点插入过程中的详细信息，便于问题排查。
    *   `fuse_bn` (`bool`):
        指示在准备过程中是否应自动执行 Conv/Linear -> BatchNorm 的融合操作。
        如果为 `True`，`prepare` 方法会调用 `graph_utils.fuse_conv_linear_bn_fx`
        来优化模型结构。

    **使用场景与上下文:**
    `Quantizer` 通常是整个量化工作流的第一步（准备阶段）。
    *   **对于 PTQ (Post-Training Quantization):**
        1.  **准备:** 使用 `Quantizer(qconfig).prepare(model)` 得到 `prepared_model`。
        2.  **校准:** 将 `prepared_model` 设置为 `eval()` 模式，通过 `enable_observer(True)`
            和 `enable_fake_quant(False)` 激活 `FakeQuantize` 中的 Observer。然后用
            代表性的校准数据集运行 `prepared_model`，让 Observer 收集统计信息。
        3.  **计算参数:** 对 `prepared_model` 中的每个 `FakeQuantize` 调用
            `calculate_qparams()`，将收集到的统计信息转换为固定的 `scale` 和 `zero_point`。
        4.  **转换/评估:** 可以选择性地将 `prepared_model` 转换为真正的量化模型
            (例如，替换为 `nn.quantized` 模块)，或者在 `eval()` 模式下保持
            `FakeQuantize` 启用 (`enable_fake_quant(True)`) 来评估模拟量化后的精度。
    *   **对于 QAT (Quantization-Aware Training):**
        1.  **准备:** 使用 `Quantizer(qconfig).prepare(model)` 得到 `prepared_model`。
        2.  **进入 QAT 模式:** 对 `prepared_model` 中的每个 `FakeQuantize` 调用
            `enable_qat(True, ...)`，启用伪量化，根据 QAT 模式配置 Observer 和
            `scale`/`zp` 的学习状态。
        3.  **训练:** 像训练普通模型一样训练 `prepared_model`。`FakeQuantize` 会模拟
            量化效应，并通过 STE 进行梯度反传。
        4.  **转换/评估:** 训练完成后，可以将模型转换为真正的量化模型或进行评估。

    **总结:** `Quantizer` 是一个利用 `torch.fx` 实现的、高度自动化的模型准备工具，
    它通过智能地插入 `FakeQuantize` 节点，为后续的 PTQ 校准或 QAT 训练铺平了道路，
    极大地简化了模型量化的准备工作。
    """

    def __init__(self, qconfig: QConfig, debug: bool = False, fuse_bn: bool = False):
        """
        初始化 Quantizer。

        Args:
            qconfig (Union[QConfig, QConfigMapping]): 量化配置。可以是应用于所有可量化
                层的单个 QConfig，也可以是用于更精细控制的 QConfigMapping。
            debug (bool, optional): 是否启用详细日志输出。默认为 False。
            fuse_bn (bool, optional): 是否在 prepare 阶段自动融合 Conv/Linear 和 BatchNorm。
                                     默认为 True。
        """
        if not isinstance(qconfig, QConfig):
            raise ValueError("qconfig 必须是 QConfig 的实例")
        if qconfig.activation is not None and not callable(qconfig.activation):
            raise TypeError(
                f"qconfig.activation 必须是可调用的工厂函数或 None，但得到的是 {type(qconfig.activation)}"
            )
        if qconfig.weight is not None and not callable(qconfig.weight):
            raise TypeError(
                f"qconfig.weight 必须是可调用的工厂函数或 None，但得到的是 {type(qconfig.weight)}"
            )

        self.qconfig = qconfig
        self.debug = debug
        self.fuse_bn = fuse_bn
        self._insertion_point_counter = 0
        self.logger = logger
        # 设置此模块和 graph_utils 模块的日志级别
        log_level = logging.DEBUG if self.debug else logging.INFO
        for h in self.logger.handlers:
            h.setLevel(log_level)
        # 也设置 graph_utils 的级别
        for h in graph_utils_logger.handlers:
            h.setLevel(log_level)
        # 确保 graph_utils 的 logger 级别也被设置 (如果 quantizer debug=True, graph_utils 也 debug)
        graph_utils_logger.setLevel(log_level)

        if self.debug:
            self.logger.debug(
                _colorize(
                    f"[{'调试模式'}] Quantizer 初始化 (BN融合: {'启用' if self.fuse_bn else '禁用'})",
                    COLOR_BOLD + COLOR_CYAN,
                )
            )
        else:
            self.logger.info(
                _colorize(
                    f"[{'标准模式'}] Quantizer 初始化 (BN融合: {'启用' if self.fuse_bn else '禁用'})",
                    COLOR_BOLD + COLOR_CYAN,
                )
            )

    def _get_unique_module_name(self, prefix: str) -> str:
        """生成唯一的模块名称以避免冲突"""
        name = f"{prefix}_{self._insertion_point_counter}"
        self._insertion_point_counter += 1
        return name

    def prepare(self, model: nn.Module) -> GraphModule:
        """
        准备模型以进行量化。
        包括可选的 BN 融合和插入 FakeQuantize 节点。
        """
        self.logger.info(
            _colorize(
                f"{'='*20} 量化准备阶段 {'='*20}",
                COLOR_PHASE,
            )
        )

        # --- 模型追踪 ---
        try:
            self.logger.debug(_colorize("--> [步骤 1] 追踪模型...", COLOR_INFO))
            original_training_state = model.training
            model.eval()  # 追踪和融合都需要 eval 模式

            tracer = fx.Tracer()
            graph = tracer.trace(model)
            # 创建副本进行修改，不影响原始模型
            model_copy = copy.deepcopy(model)
            # 将模型及其图包装在 GraphModule 中
            # 注意：此时 model_copy 仍处于 eval 模式
            prepared_model = GraphModule(model_copy, graph)

            self.logger.info(_colorize("[成功] 模型追踪完成！", COLOR_SUCCESS))
            if self.debug:
                self.logger.debug(
                    _colorize("\n--- 初始 FX 图 (追踪后) ---", COLOR_BOLD + COLOR_INFO)
                )
                graph.print_tabular()
                self.logger.debug(
                    _colorize("--- (结束初始图) ---\n", COLOR_BOLD + COLOR_INFO)
                )
            # 保持 eval 模式进入融合步骤

        except Exception as e:
            model.train(original_training_state)  # 出错时恢复状态
            self.logger.exception(_colorize(f"[错误] 模型追踪失败: {e}", COLOR_ERROR))
            raise RuntimeError(f"模型追踪失败: {e}") from e

        graph = prepared_model.graph
        modules = dict(prepared_model.named_modules())

        # --- （可选）BN 融合 ---
        modules_to_fuse = []  # 重置，以防 prepare 被多次调用
        fusion_executed = False  # 标记是否实际执行了融合调用
        if self.fuse_bn:
            self.logger.info(
                _colorize(
                    f"{'='*20} BN 融合 {'='*20}",
                    COLOR_PHASE,
                )
            )
            try:
                graph = prepared_model.graph
                modules = dict(prepared_model.named_modules())  # 获取当前模型副本的模块

                supported_convs = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
                supported_linears = (nn.Linear,)
                supported_bns = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

                self.logger.debug(
                    _colorize("--> 搜索 Conv/Linear -> BN 模式...", COLOR_INFO)
                )
                for node in graph.nodes:
                    if node.op == "call_module" and isinstance(
                        modules.get(str(node.target)),
                        (supported_convs + supported_linears),
                    ):
                        conv_or_linear_node = node
                        # 检查用户是否 **唯一** 且为 BN
                        if len(conv_or_linear_node.users) == 1:
                            user_node = next(iter(conv_or_linear_node.users))
                            if user_node.op == "call_module" and isinstance(
                                modules.get(str(user_node.target)), supported_bns
                            ):
                                bn_node = user_node
                                bn_module = modules.get(str(bn_node.target))
                                # 确保 BN 模块有运行统计数据 (在 eval 模式下通常是有的)
                                if (
                                    hasattr(bn_module, "running_mean")
                                    and hasattr(bn_module, "running_var")
                                    and bn_module.running_mean is not None
                                    and bn_module.running_var is not None
                                ):
                                    module_pair = [
                                        str(conv_or_linear_node.target),
                                        str(bn_node.target),
                                    ]
                                    modules_to_fuse.append(module_pair)
                                    self.logger.debug(
                                        _colorize(
                                            f"  发现可融合对: {_colorize(module_pair[0], COLOR_MODULE)} -> {_colorize(module_pair[1], COLOR_MODULE)}",
                                            COLOR_DEBUG,
                                        )
                                    )
                                else:
                                    self.logger.debug(
                                        _colorize(
                                            f"  跳过融合检测: BN 模块 {_colorize(str(bn_node.target), COLOR_MODULE)} 缺少运行统计数据。确保模型已训练并在 eval() 模式。",
                                            COLOR_WARN,
                                        )
                                    )
                            # else: # 如果用户不是 BN 模块，则不融合
                        # else: # 如果有多个用户，则不融合

                if modules_to_fuse:
                    self.logger.info(
                        _colorize(
                            f"找到 {len(modules_to_fuse)} 对 Conv/Linear-BN 可进行融合。调用融合函数...",
                            COLOR_INFO,
                        )
                    )
                    # --- 执行融合 ---
                    # 调用 graph_utils.py 中的函数
                    # 它会直接修改 prepared_model
                    prepared_model = fuse_conv_linear_bn_fx(
                        prepared_model, modules_to_fuse
                    )
                    fusion_executed = True  # 标记已调用融合

                    # --- 关键：融合函数内部会处理图和模块的修改，并重新编译 ---
                    # 所以这里只需要更新本地的 graph 和 modules 引用以进行后续处理
                    graph = prepared_model.graph
                    modules = dict(
                        prepared_model.named_modules()
                    )  # 获取融合后的模块字典
                    self.logger.info(_colorize("[成功] BN 融合完成！", COLOR_SUCCESS))
                    if self.debug:
                        self.logger.debug(
                            _colorize(
                                "\n--- FX 图 (BN 融合后) ---", COLOR_BOLD + COLOR_INFO
                            )
                        )
                        # 融合函数内部可能已经 recompile 过，所以图应该是最新的
                        graph.print_tabular()
                        self.logger.debug(
                            _colorize("--- (结束融合图) ---\n", COLOR_BOLD + COLOR_INFO)
                        )
                else:
                    self.logger.info(
                        _colorize("未找到可融合的 Conv/Linear-BN 对。", COLOR_INFO)
                    )

            except Exception as e:
                # 如果融合失败，记录错误但可能继续（取决于策略）
                model.train(original_training_state)  # 恢复原始模型状态
                self.logger.exception(
                    _colorize(f"[错误] BN 融合失败: {e}", COLOR_ERROR)
                )
                # 这里选择继续，使用未融合的图
                # raise RuntimeError(f"BN 融合失败: {e}") from e # 如果希望融合失败阻止后续步骤
                self.logger.warning(
                    _colorize(
                        "BN 融合失败，将继续进行量化器插入（使用追踪后的原始图）。",
                        COLOR_WARN,
                    )
                )
                # 确保 graph 和 modules 是融合前的状态 (从 prepared_model 获取)
                graph = prepared_model.graph
                modules = dict(prepared_model.named_modules())
                fusion_executed = False  # 融合未成功执行

        # --- 量化器（FakeQuantize）插入 ---
        # 使用融合（或未融合）后的 graph 和 modules
        self._insertion_point_counter = 0  # 重置插入计数器
        nodes_to_process = list(graph.nodes)  # 基于当前图状态获取节点列表
        num_nodes = len(nodes_to_process)
        self.logger.debug(
            _colorize(
                f"\n--> [步骤 2] 处理 {num_nodes} 个图节点以插入量化器 (基于{'融合后' if fusion_executed else ('未融合' if self.fuse_bn else '原始')}图)...",
                COLOR_INFO,
            )
        )

        self._printed_no_activation_config = False

        # !!! 节点遍历和量化器插入 !!!
        # 作用于 `prepared_model` 的当前状态（可能是融合后的）
        for i, node in enumerate(nodes_to_process):
            if self.debug:
                op_color = COLOR_OPERATOR
                target_color = COLOR_OPERATOR
                target_str = str(node.target)
                node_type_info = ""  # 用于显示模块类型
                if node.op == "call_module":
                    op_color = COLOR_MODULE
                    target_color = COLOR_MODULE
                    # 检查模块是否存在于当前 modules 字典中
                    if target_str in modules:
                        target_module = modules[target_str]
                        node_type_info = (
                            f" ({type(target_module).__name__})"  # 显示类型
                        )
                        target_str = (
                            _colorize(target_str, target_color) + node_type_info
                        )
                    else:
                        # 这个模块可能已被融合删除
                        target_str = (
                            _colorize(target_str, COLOR_WARN) + " (模块可能已融合/移除)"
                        )
                elif node.op == "get_attr":
                    target_color = COLOR_TARGET
                    target_str = _colorize(target_str, target_color)
                elif node.op == "placeholder":
                    target_color = COLOR_INPUT
                    target_str = _colorize(target_str, target_color)
                elif node.op == "output":
                    target_color = COLOR_TARGET
                    target_str = _colorize(
                        "output", target_color
                    )  # 输出节点没有 target 属性
                else:  # call_function, call_method
                    target_str = _colorize(target_str, target_color)

                node_info = f"节点 {i+1:>{len(str(num_nodes))}}/{num_nodes}: "
                node_details = (
                    f"{_colorize(node.name, COLOR_NODE)} "
                    f"(op={_colorize(node.op, op_color)}, "
                    f"target={target_str})"
                )
                self.logger.debug(
                    _colorize(f"\n{node_info}{node_details}", COLOR_DEBUG)
                )

            # --- 权重量化器插入 ---
            if node.op == "call_module":
                target_key = str(node.target)
                # 使用更新后的 modules 字典检查
                if target_key not in modules:
                    # 如果模块不在字典里（例如，被融合掉的 BN），跳过
                    if self.debug:
                        self.logger.debug(
                            _colorize(
                                f"  [信息][{_colorize('权重', COLOR_TARGET)}] 模块 '{_colorize(target_key, COLOR_MODULE)}' 不在当前模块字典中。跳过权重检查。",
                                COLOR_DEBUG,
                            )
                        )
                    continue  # 跳过处理不存在的模块

                target_module = modules.get(target_key)  # 使用更新后的 modules

                # 检查的是融合后的模块 (例如，Conv2d 是否需要量化权重)
                if (
                    target_module
                    and is_quantizable_weight_module(target_module)
                    and self.qconfig.weight
                    and hasattr(target_module, "weight")  # 融合后的层肯定有 weight
                    and target_module.weight is not None
                ):
                    # 从 QConfig 获取 FakeQuantize 工厂函数
                    weight_quant_factory = self.qconfig.weight
                    # 调用工厂创建 FakeQuantize 实例
                    weight_quant_instance = weight_quant_factory()
                    # 确保创建的是 FakeQuantize (或其子类)
                    if not isinstance(weight_quant_instance, FakeQuantize):
                        raise TypeError(
                            f"QConfig.weight 工厂必须返回 FakeQuantize 实例，但得到 {type(weight_quant_instance)}"
                        )
                    # 使用融合后模块的名称生成观察器名称
                    quant_module_name = self._get_unique_module_name(
                        f"weight_fake_quant_{target_key.replace('.', '_')}"
                    )
                    # 在 prepared_model (可能已融合) 上添加模块
                    prepared_model.add_module(quant_module_name, weight_quant_instance)

                    # 目标是融合后模块的权重
                    weight_attr_target = f"{target_key}.weight"
                    self.logger.debug(
                        _colorize(
                            f"  [{_colorize('插入', COLOR_ACTION)}][{_colorize('权重', COLOR_TARGET)}] 发现可量化模块: '{_colorize(target_key, COLOR_MODULE)}' ({type(target_module).__name__})。 "
                            f"为 '{_colorize(weight_attr_target, COLOR_TARGET)}' 插入量化器 '{_colorize(quant_module_name, COLOR_MODULE)}'。",
                            COLOR_DEBUG,
                        )
                    )

                    # 图操作在融合后的 graph 上进行
                    with graph.inserting_before(node):
                        # 1. 获取权重属性
                        get_attr_node = graph.get_attr(weight_attr_target)
                        # 2. 调用权重观察器
                        observer_call_node = graph.call_module(
                            quant_module_name, args=(get_attr_node,)
                        )
                        # 3. 将原始模块节点对权重的引用替换为观察器的输出
                        # 注意：这里修改的是 node (e.g., conv2d) 的 *输入参数列表* 中
                        # 对 `target_key.weight` 的引用，而不是直接替换 get_attr 节点。
                        # 这需要知道 weight 参数在 call_module 中的位置，通常是第一个非 self 参数。
                        # 更健壮的方式是使用 node.replace_input_with(old_node, new_node)
                        # 这里我们假设 get_attr_node 就是那个旧输入。
                        node.replace_input_with(get_attr_node, observer_call_node)

                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('get_attr', COLOR_OPERATOR)}: '{_colorize(get_attr_node.name, COLOR_NODE)}' (获取 '{weight_attr_target}')",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('call_module', COLOR_MODULE)} (权重 量化器): '{_colorize(observer_call_node.name, COLOR_NODE)}' (调用 '{quant_module_name}')",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 更新 '{_colorize(node.name, COLOR_NODE)}' 的输入，用 '{_colorize(observer_call_node.name, COLOR_NODE)}' 替换 '{_colorize(get_attr_node.name, COLOR_NODE)}'。",
                            COLOR_DEBUG,
                        )
                    )
                elif (
                    self.debug
                    and target_module
                    and is_quantizable_weight_module(target_module)
                ):
                    reason = ""
                    if not self.qconfig.weight:
                        reason = "QConfig 未配置权重量化器"
                    elif not hasattr(target_module, "weight"):
                        reason = "模块无 'weight' 属性"
                    elif target_module.weight is None:
                        reason = "模块 'weight' 属性为 None"
                    else:
                        reason = "未知原因 (可能已满足其他条件)"
                    self.logger.debug(
                        _colorize(
                            f"  [跳过][{_colorize('权重', COLOR_TARGET)}] 模块 '{_colorize(target_key, COLOR_MODULE)}' ({type(target_module).__name__})。 {_colorize('原因:', COLOR_REASON)} {reason}",
                            COLOR_DEBUG,
                        )
                    )

            # --- 激活量化器插入 ---
            should_quantize_output = False
            output_prefix = ""
            reason_for_quantization = ""
            origin_node_desc = ""
            is_output_from_quant_module = False  # 标记当前节点是否就是个量化器

            # 检查当前节点是否已经是量化器调用
            if node.op == "call_module":
                target_key = str(node.target)
                if target_key in modules:
                    current_module = modules.get(target_key)
                    # 检查是否是 QConfig 中定义的激活或权重量化器实例
                    is_act_quant_inst = self.qconfig.activation and isinstance(
                        current_module, FakeQuantize
                    )
                    # 权重观察器后面通常不需要再加激活观察器
                    is_wt_quant_inst = self.qconfig.weight and isinstance(
                        current_module, FakeQuantize
                    )
                    if is_act_quant_inst or is_wt_quant_inst:
                        is_output_from_quant_module = True
                        if self.debug:
                            self.logger.debug(
                                _colorize(
                                    f"  [信息][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 本身是量化器调用 ('{_colorize(target_key, COLOR_MODULE)}').",
                                    COLOR_DEBUG,
                                )
                            )

            # 决定是否需要量化当前节点的输出 (如果它不是量化器本身)
            if not is_output_from_quant_module and self.qconfig.activation:
                target_key = str(node.target)  # 重新获取以防万一
                if node.op == "call_module":
                    # 使用更新后的 modules 检查
                    if target_key in modules:
                        target_module = modules.get(target_key)
                        # 检查融合后的模块 (或其他模块) 是否需要激活量化
                        if target_module and is_quantizable_activation_module(
                            target_module
                        ):
                            should_quantize_output = True
                            target_str_safe = target_key.replace(".", "_")
                            output_prefix = (
                                f"act_fake_quant_after_mod_{target_str_safe}"
                            )
                            origin_node_desc = f"模块 '{_colorize(target_key, COLOR_MODULE)}' ({type(target_module).__name__})"
                            reason_for_quantization = f"{origin_node_desc} 的输出"
                    # else: # 如果模块不在字典里（如被融合的 BN），自然不需要为其输出加观察器
                elif node.op == "call_function":
                    # hasattr 检查是为了防止 node.target 是字符串或其他非 callable 对象
                    if callable(node.target) and is_quantizable_activation_function(
                        node.target
                    ):
                        should_quantize_output = True
                        func_name = getattr(node.target, "__name__", str(node.target))
                        output_prefix = f"act_fake_quant_after_func_{func_name}"
                        origin_node_desc = (
                            f"函数 '{_colorize(func_name, COLOR_OPERATOR)}'"
                        )
                        reason_for_quantization = f"{origin_node_desc} 的输出"
                    # elif self.debug and callable(node.target): # 不可量化的function的debug信息
                    #     self.logger.debug(...)
                elif node.op == "call_method":
                    method_name = target_key  # 对于 call_method, target 是方法名字符串
                    if is_quantizable_activation_method(method_name):
                        should_quantize_output = True
                        output_prefix = f"act_fake_quant_after_method_{method_name}"
                        origin_node_desc = (
                            f"方法 '{_colorize(method_name, COLOR_OPERATOR)}'"
                        )
                        reason_for_quantization = f"{origin_node_desc} 的输出"
                    # elif self.debug: # 不可量化的method的debug信息
                    #      self.logger.debug(...)
                elif node.op == "placeholder":
                    # 输入占位符通常需要量化，除非其用户已经是量化器
                    already_quantized_by_user = False
                    for user in node.users:
                        user_target_key = str(user.target)
                        if user.op == "call_module" and user_target_key in modules:
                            user_module = modules.get(user_target_key)
                            if self.qconfig.activation and isinstance(
                                user_module, FakeQuantize
                            ):
                                already_quantized_by_user = True
                                if self.debug:
                                    self.logger.debug(
                                        _colorize(
                                            f"  [跳过][{_colorize('激活', COLOR_TARGET)}] 输入 '{_colorize(node.name, COLOR_INPUT)}' 已被用户 '{_colorize(user.name, COLOR_NODE)}' ({_colorize(user_target_key, COLOR_MODULE)}) 观察。",
                                            COLOR_DEBUG,
                                        )
                                    )
                                break  # 只要有一个用户是观察器就够了
                    if not already_quantized_by_user:
                        should_quantize_output = True
                        # node.target 是占位符的名称 (例如 'x')
                        output_prefix = f"act_fake_quant_input_{node.target}"
                        origin_node_desc = (
                            f"输入 '{_colorize(node.target, COLOR_INPUT)}'"
                        )
                        reason_for_quantization = f"{origin_node_desc} 的值"

            # 插入激活量化器（如果需要且 QConfig 配置了）
            if should_quantize_output and self.qconfig.activation:
                # 再次检查：这个节点的输出是否已经被某个下游节点观察了
                # (这主要处理 M 输出 -> 多个 Op，其中一个 Op 是 Quant 的情况)
                # 但更常见的是在 placeholder 检查中处理输入观察。
                # 对于中间节点，我们通常在其 *之后* 插入观察器。
                # 这个检查主要是为了防止在已经有观察器的地方重复插入。
                is_already_quantized_by_user = False
                for user in node.users:
                    user_target_key = str(user.target)
                    if user.op == "call_module" and user_target_key in modules:
                        user_module = modules.get(user_target_key)
                        # 如果用户节点本身就是激活观察器
                        if isinstance(user_module, FakeQuantize):
                            is_already_quantized_by_user = True
                            if self.debug:
                                self.logger.debug(
                                    _colorize(
                                        f"  [跳过][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 的输出 ({reason_for_quantization}) 已被后续用户 "
                                        f"'{_colorize(user.name, COLOR_NODE)}' ('{_colorize(user_target_key, COLOR_MODULE)}') 观察。",
                                        COLOR_DEBUG,
                                    )
                                )
                            break

                if not is_already_quantized_by_user:
                    act_quant_factory = self.qconfig.activation
                    act_quant_instance = act_quant_factory()
                    if not isinstance(act_quant_instance, FakeQuantize):
                        raise TypeError(
                            f"QConfig.activation 工厂必须返回 FakeQuantize 实例，但得到 {type(act_quant_instance)}"
                        )
                    quant_module_name = self._get_unique_module_name(output_prefix)
                    prepared_model.add_module(quant_module_name, act_quant_instance)

                    self.logger.debug(
                        _colorize(
                            f"  [{_colorize('插入', COLOR_ACTION)}][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 的输出需要观察。 "
                            f"{_colorize('原因:', COLOR_REASON)} {reason_for_quantization}. "
                            f"插入量化器 '{_colorize(quant_module_name, COLOR_MODULE)}'。",
                            COLOR_DEBUG,
                        )
                    )

                    # 在当前节点之后插入量化器调用
                    with graph.inserting_after(node):
                        inserted_node = graph.call_module(
                            quant_module_name, args=(node,)  # 观察器的输入是当前节点
                        )

                    # 将原节点的所有用户（除了新插入的观察器节点本身）
                    # 的输入从原节点更新为新插入的观察器节点
                    users_to_update = list(node.users.keys())  # 创建副本以安全迭代
                    updated_users_count = 0
                    for user_node in users_to_update:
                        if user_node != inserted_node:  # 不要更新观察器节点自己
                            user_node.replace_input_with(node, inserted_node)
                            updated_users_count += 1
                    # 确保插入节点的参数是正确的 (虽然 insert_after 通常会做)
                    inserted_node.args = (node,)

                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('call_module', COLOR_MODULE)} (激活 量化器): '{_colorize(inserted_node.name, COLOR_NODE)}' (调用 '{quant_module_name}')",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 更新了 '{_colorize(node.name, COLOR_NODE)}' 的 {updated_users_count} 个下游用户，使其使用 '{_colorize(inserted_node.name, COLOR_NODE)}'。",
                            COLOR_DEBUG,
                        )
                    )
            elif (
                self.debug
                and not should_quantize_output  # 不需要量化输出
                and not is_output_from_quant_module  # 且节点本身不是量化器
                and node.op != "output"  # 且不是最终输出节点
            ):
                if not self.qconfig.activation:  # 如果是因为没配置激活量化器
                    if not self._printed_no_activation_config:  # 只打印一次
                        self.logger.debug(
                            _colorize(
                                f"  [信息][{_colorize('激活', COLOR_TARGET)}] 跳过所有激活量化器插入。{_colorize('原因:', COLOR_REASON)} QConfig 未配置激活量化器。",
                                COLOR_INFO,
                            )
                        )
                        self._printed_no_activation_config = True
                # else: # 如果配置了，但当前节点不满足条件 (e.g., 非量化类型)
                #     self.logger.debug(...) # 可以选择性地记录为什么这个特定节点跳过

        # --- 清理和重新编译 ---
        try:
            self.logger.debug(
                _colorize(
                    "\n--> [步骤 3] 校验图 (检查有效性)...",
                    COLOR_INFO,
                )
            )
            graph.lint()  # 检查图的结构是否有效
            prepared_model = GraphModule(prepared_model, graph)  # 重新创建 GraphModule
            self.logger.debug(_colorize("[成功] 图校验通过", COLOR_SUCCESS))
        except Exception as e:
            # 这里也使用 logger.exception
            self.logger.exception(
                _colorize(f"[错误] 插入后图校验失败: {e}", COLOR_ERROR)
            )
            if self.debug:
                # 在调试模式下出错时保留图转储
                self.logger.debug(
                    _colorize(
                        "\n--- 校验错误前的图状态 ---",
                        COLOR_BOLD + COLOR_ERROR,
                    )
                )
                try:
                    graph.print_tabular()  # 为了调试方便，仍然直接打印
                except Exception as pe:
                    self.logger.error(_colorize(f"(打印图失败: {pe})", COLOR_ERROR))
                self.logger.debug(
                    _colorize("--- (结束图状态) ---\n", COLOR_BOLD + COLOR_ERROR)
                )
            raise RuntimeError(f"插入后图校验失败: {e}") from e

        if self.debug:
            self.logger.debug(
                _colorize(
                    "\n--- 最终 FX 图 (准备阶段完成) ---",
                    COLOR_BOLD + COLOR_INFO,
                )
            )
            prepared_model.graph.print_tabular()
            self.logger.debug(
                _colorize("--- (结束最终图) ---\n", COLOR_BOLD + COLOR_INFO)
            )

        # 恢复原始模型的训练状态 (虽然返回的是副本)
        model.train(original_training_state)

        self.logger.info(
            _colorize(
                f"{'='*20} 量化准备阶段完成 {'='*20}",
                COLOR_PHASE + COLOR_SUCCESS,
            )
        )
        fusion_status = "未启用BN融合"
        if self.fuse_bn:
            fusion_status = (
                "已执行BN融合" if fusion_executed else "尝试BN融合但未找到可融合对"
            )

        self.logger.info(
            _colorize(
                f"模型准备完毕 ({fusion_status}, 已插入权重和激活量化器)。",
                COLOR_SUCCESS,
            )
        )
        return prepared_model  # 返回准备好的模型副本
