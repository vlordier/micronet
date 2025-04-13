import copy
import logging
import sys

import torch.nn as nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule

from .qconfig import QConfig
from .graph_utils import (
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
)

# --- 配置此模块的日志记录器 ---
# 使用特定的名称以避免冲突
logger = logging.getLogger("micronet.quantizer")
# 防止 Quantizer 被多次实例化时出现重复的处理器
if not logger.hasHandlers():
    # 默认处理器：StreamHandler 输出到控制台 (stderr)
    handler = logging.StreamHandler(sys.stderr)  # 日志通常使用 stderr
    # 基本格式化器，因为着色在消息字符串内部处理
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # 将日志记录器的基本级别设置为 DEBUG 以捕获所有信息
    logger.setLevel(logging.DEBUG)
    # --- 关键：如果根日志记录器在别处配置，避免将消息传播到根日志记录器 ---
    # logger.propagate = False # 如果您有全局日志设置并且想要隔离此日志记录器，请取消注释此行


class Quantizer:
    """
    管理使用 torch.fx 对 PyTorch 模型进行量化准备过程的核心类。

    此类负责接收一个原始的 nn.Module 模型和一个 QConfig 量化配置对象。
    它的主要功能是通过符号追踪（Symbolic Tracing）获取模型的计算图（FX Graph），
    然后根据 QConfig 中的设定，在图中的适当位置（例如，特定模块的权重之前，
    或特定操作/模块的输出之后）插入量化节点（通常是 Observer 或 FakeQuantize 模块）。

    主要工作流程集中在 `prepare` 方法中，该方法执行以下操作：
    1. 使用 `torch.fx` 追踪原始模型，生成计算图。
    2. 遍历计算图中的节点。
    3. 根据节点的类型（调用模块、函数、方法、占位符）和 `QConfig` 的配置，
       判断是否需要插入权重量化器或激活量化器。
    4. 在图中插入相应的量化模块（来自 `QConfig`）。
    5. 返回一个修改后的、包含量化节点的 `torch.fx.GraphModule`。

    这个准备好的模型随后可以用于：
    - 后训练量化（PTQ）的校准（Calibration）阶段：运行数据通过模型以收集统计信息。
    - 量化感知训练（QAT）：在训练过程中模拟量化效应。

    Attributes:
        qconfig (QConfig): 存储量化配置，指定用于权重和激活的量化器模块工厂。
        debug (bool): 控制是否启用详细的调试日志输出。
        logger (logging.Logger): 用于记录量化过程信息的日志记录器。
    """

    def __init__(self, qconfig: QConfig, debug: bool = False):
        """
        初始化 Quantizer 实例。

        Args:
            qconfig (QConfig): 一个 QConfig 实例，定义了用于权重和激活的
                               量化器/观察器模块的工厂函数。此配置将指导
                               `prepare` 方法中插入哪些类型的量化节点。
                               `qconfig.activation` 和 `qconfig.weight`
                               必须是可调用的工厂函数（例如类）或 None。
            debug (bool, optional): 如果为 True，将启用详细的调试日志记录，
                                    包括每个节点的处理、插入决策和图的中间状态。
                                    默认为 False，只记录关键信息和阶段转换。

        Raises:
            ValueError: 如果提供的 `qconfig` 不是 `QConfig` 的实例。
            TypeError: 如果 `qconfig.activation` 或 `qconfig.weight`
                       不是可调用的工厂函数或 None。

        主要作用：
        - 验证输入的 `qconfig` 是否有效。
        - 存储 `qconfig` 和 `debug` 标志。
        - 初始化用于生成唯一量化模块名称的内部计数器。
        - 获取并配置模块级别的日志记录器，根据 `debug` 标志设置日志级别。
        - 记录一条初始化消息，指明是在调试模式还是标准模式下运行。
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
        self._insertion_point_counter = 0

        # --- 使用模块级别的日志记录器 ---
        self.logger = logger
        # --- 根据 debug 标志设置处理器的级别 ---
        # 这控制了实际输出到控制台的内容
        for h in self.logger.handlers:
            h.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # 使用 logger.debug 输出初始消息
        self.logger.debug(
            _colorize(
                f"[{'调试模式' if self.debug else '标准模式'}] 量化模型初始化",  # 在非调试模式下也提供信息
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
        准备模型以进行量化（PTQ 校准或 QAT 训练）。
        为激活和权重插入观察器/占位符。
        """
        # 使用 logger.info 标记阶段（在两种模式下都可见）
        self.logger.info(
            _colorize(
                f"\n{'='*20} 量化准备阶段 {'='*20}",
                COLOR_PHASE,
            )
        )

        # --- 模型追踪 ---
        try:
            self.logger.debug(_colorize("--> [步骤 1] 追踪模型...", COLOR_INFO))
            original_training_state = model.training
            model.eval()
            tracer = fx.Tracer()
            graph = tracer.trace(model)
            model_copy = copy.deepcopy(model)
            traced_model = GraphModule(model_copy, graph)

            # 使用 logger.info 标记主要成功（在两种模式下都可见）
            self.logger.info(_colorize("[成功] 模型追踪完成！", COLOR_SUCCESS))
            if self.debug:
                self.logger.debug(
                    _colorize("\n--- 初始 FX 图 ---", COLOR_BOLD + COLOR_INFO)
                )
                # 重定向 print_tabular 的输出（它输出到 stdout）- 不太理想但必要
                # 或者，如果需要，捕获它并记录下来，但直接打印对于调试通常没问题
                # 目前，将 print_tabular 保留在调试检查内
                graph.print_tabular()
                self.logger.debug(
                    _colorize("--- (结束初始图) ---\n", COLOR_BOLD + COLOR_INFO)
                )
            model.train(original_training_state)
        except Exception as e:
            # 使用 logger.exception 记录错误以自动包含回溯信息
            self.logger.exception(_colorize(f"[错误] 模型追踪失败: {e}", COLOR_ERROR))
            raise RuntimeError(f"模型追踪失败: {e}") from e

        graph = traced_model.graph
        modules = dict(traced_model.named_modules())
        self._insertion_point_counter = 0

        nodes_to_process = list(graph.nodes)
        num_nodes = len(nodes_to_process)
        self.logger.debug(
            _colorize(
                f"\n--> [步骤 2] 处理 {num_nodes} 个图节点以插入量化器...",
                COLOR_INFO,
            )
        )

        self._printed_no_activation_config = False

        for i, node in enumerate(nodes_to_process):
            if self.debug:
                # 节点处理细节是调试级别
                op_color = COLOR_OPERATOR
                target_color = COLOR_OPERATOR
                target_str = str(node.target)
                if node.op == "call_module":
                    op_color = COLOR_MODULE
                    target_color = COLOR_MODULE
                    target_str = _colorize(target_str, target_color)
                elif node.op == "get_attr":
                    target_color = COLOR_TARGET
                    target_str = _colorize(target_str, target_color)
                elif node.op == "placeholder":
                    target_color = COLOR_INPUT
                    target_str = _colorize(target_str, target_color)
                else:
                    target_str = _colorize(target_str, target_color)

                node_info = f"节点 {i+1:>{len(str(num_nodes))}}/{num_nodes}: "
                node_details = (
                    f"{_colorize(node.name, COLOR_NODE)} "
                    f"(操作={_colorize(node.op, op_color)}, "
                    f"目标={target_str})"
                )
                self.logger.debug(
                    _colorize(f"\n{node_info}{node_details}", COLOR_DEBUG)
                )

            # --- 动态权重 量化器 插入 ---
            if node.op == "call_module":
                target_key = str(node.target)
                if target_key not in modules:
                    # 对意外情况（如未找到模块）使用 logger.warning
                    self.logger.warning(
                        _colorize(
                            f"  [跳过][{_colorize('权重', COLOR_TARGET)}] 模块 '{_colorize(target_key, COLOR_MODULE)}' 在追踪的模型模块中未找到。",
                            COLOR_WARN,
                        )
                    )
                    continue

                target_module = modules.get(target_key)

                if (
                    target_module
                    and is_quantizable_weight_module(target_module)
                    and self.qconfig.weight
                    and hasattr(target_module, "weight")
                    and target_module.weight is not None
                ):
                    weight_quant_module_cls = self.qconfig.weight
                    weight_quant_instance = weight_quant_module_cls()
                    quant_module_name = self._get_unique_module_name(
                        f"weight_obs_{target_key.replace('.', '_')}"
                    )
                    traced_model.add_module(quant_module_name, weight_quant_instance)

                    weight_attr_target = f"{target_key}.weight"
                    # 插入信息是 DEBUG 级别
                    self.logger.debug(
                        _colorize(
                            f"  [{_colorize('插入', COLOR_ACTION)}][{_colorize('权重', COLOR_TARGET)}] 发现可量化模块: '{_colorize(target_key, COLOR_MODULE)}'。 "
                            f"为 '{_colorize(weight_attr_target, COLOR_TARGET)}' 插入量化器 '{_colorize(quant_module_name, COLOR_MODULE)}'。",
                            COLOR_DEBUG,
                        )
                    )

                    with graph.inserting_before(node):
                        get_attr_node = graph.get_attr(weight_attr_target)
                        observer_call_node = graph.call_module(
                            quant_module_name, args=(get_attr_node,)
                        )
                        # FX会自动处理输入替换，这里显式调用以确保清晰
                        node.replace_input_with(get_attr_node, observer_call_node)

                    # 详细的插入步骤是 DEBUG 级别
                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('get_attr', COLOR_OPERATOR)}: '{_colorize(get_attr_node.name, COLOR_NODE)}'",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('call_module', COLOR_MODULE)} (权重 量化器): '{_colorize(observer_call_node.name, COLOR_NODE)}'",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 更新 '{_colorize(node.name, COLOR_NODE)}' 的输入以使用量化器输出。",
                            COLOR_DEBUG,
                        )
                    )

                elif (
                    self.debug  # 仅在调试模式下记录跳过原因
                    and target_module
                    and is_quantizable_weight_module(target_module)
                ):
                    reason = ""
                    if not self.qconfig.weight:
                        reason = "QConfig 没有配置权重量化器 (`qconfig.weight` 为 None)"
                    elif not hasattr(target_module, "weight"):
                        reason = f"模块没有 'weight' 属性"
                    elif target_module.weight is None:
                        reason = f"模块 'weight' 属性为 None"
                    # 由于配置或正常原因跳过是 DEBUG 级别
                    self.logger.debug(
                        _colorize(
                            f"  [跳过][{_colorize('权重', COLOR_TARGET)}] 模块 '{_colorize(target_key, COLOR_MODULE)}' 是可量化类型，但跳过量化器插入。 "
                            f"{_colorize('原因:', COLOR_REASON)} {reason}",
                            COLOR_DEBUG,
                        )
                    )
                # 跳过不可量化的权重模块非常冗长，保持为 DEBUG
                # elif self.debug:
                #     self.logger.debug(
                #         _colorize(
                #             f"  [信息][{_colorize('权重', COLOR_TARGET)}] 模块 '{_colorize(target_key, COLOR_MODULE)}' 不是权重可量化的类型。",
                #             COLOR_DEBUG,
                #         )
                #     )

            # --- 激活 量化器 插入 ---
            should_quantize_output = False
            output_prefix = ""
            reason_for_quantization = ""
            origin_node_desc = ""
            is_output_from_quant_module = False

            if node.op == "call_module":
                target_key = str(node.target)
                if target_key in modules:
                    current_module = modules.get(target_key)
                    is_act_obs_inst = self.qconfig.activation and isinstance(
                        current_module, self.qconfig.activation
                    )
                    is_wt_obs_inst = self.qconfig.weight and isinstance(
                        current_module, self.qconfig.weight
                    )
                    if is_act_obs_inst or is_wt_obs_inst:
                        is_output_from_quant_module = True
                        # 跳过的解释是 DEBUG 级别
                        self.logger.debug(
                            _colorize(
                                f"  [信息][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 已经是量化器调用 "
                                f"('{_colorize(target_key, COLOR_MODULE)}')，在其后不需要量化器。",
                                COLOR_DEBUG,
                            )
                        )

            if not is_output_from_quant_module and self.qconfig.activation:
                target_key = str(node.target)
                if node.op == "call_module":
                    if target_key in modules:
                        target_module = modules.get(target_key)
                        if target_module and is_quantizable_activation_module(
                            target_module
                        ):
                            should_quantize_output = True
                            target_str = target_key.replace(".", "_")
                            output_prefix = f"act_obs_after_mod_{target_str}"
                            origin_node_desc = (
                                f"可量化模块 '{_colorize(target_key, COLOR_MODULE)}'"
                            )
                            reason_for_quantization = f"{origin_node_desc} 的输出"
                elif node.op == "call_function":
                    if callable(node.target) and is_quantizable_activation_function(
                        node.target
                    ):
                        should_quantize_output = True
                        func_name = getattr(node.target, "__name__", target_key)
                        output_prefix = f"act_obs_after_func_{func_name}"
                        origin_node_desc = (
                            f"可量化函数 '{_colorize(func_name, COLOR_OPERATOR)}'"
                        )
                        reason_for_quantization = f"{origin_node_desc} 的输出"
                    elif self.debug and callable(node.target):
                        # 跳过不可量化的函数是 DEBUG 信息
                        self.logger.debug(
                            _colorize(
                                f"  [信息][{_colorize('激活', COLOR_TARGET)}] 函数 '{_colorize(getattr(node.target, '__name__', target_key), COLOR_OPERATOR)}' 的输出不需要观察。",
                                COLOR_DEBUG,
                            )
                        )
                elif node.op == "call_method":
                    method_name = target_key
                    if is_quantizable_activation_method(method_name):
                        should_quantize_output = True
                        output_prefix = f"act_obs_after_method_{method_name}"
                        origin_node_desc = (
                            f"可量化方法 '{_colorize(method_name, COLOR_OPERATOR)}'"
                        )
                        reason_for_quantization = f"{origin_node_desc} 的输出"
                    elif self.debug:
                        # 跳过不可量化的方法是 DEBUG 信息
                        self.logger.debug(
                            _colorize(
                                f"  [信息][{_colorize('激活', COLOR_TARGET)}] 方法 '{_colorize(method_name, COLOR_OPERATOR)}' 的输出不需要观察。",
                                COLOR_DEBUG,
                            )
                        )
                elif node.op == "placeholder":
                    already_observed_by_user = False
                    for user in node.users:
                        user_target_key = str(user.target)
                        if user.op == "call_module" and user_target_key in modules:
                            user_module = modules.get(user_target_key)
                            if self.qconfig.activation and isinstance(
                                user_module, self.qconfig.activation
                            ):
                                already_observed_by_user = True
                                # 跳过已被观察的占位符是 DEBUG 信息
                                self.logger.debug(
                                    _colorize(
                                        f"  [跳过][{_colorize('激活', COLOR_TARGET)}] 输入占位符 '{_colorize(node.name, COLOR_INPUT)}' 已被用户节点 "
                                        f"'{_colorize(user.name, COLOR_NODE)}' ({_colorize(user_target_key, COLOR_MODULE)}) 观察。",
                                        COLOR_DEBUG,
                                    )
                                )
                                break
                    if not already_observed_by_user:
                        should_quantize_output = True
                        output_prefix = f"act_obs_input_{node.target}"
                        origin_node_desc = (
                            f"输入占位符 '{_colorize(node.target, COLOR_INPUT)}'"
                        )
                        reason_for_quantization = f"来自 {origin_node_desc} 的输入"

            if should_quantize_output and self.qconfig.activation:
                is_already_observed_by_user = False
                for user in node.users:
                    user_target_key = str(user.target)
                    if user.op == "call_module" and user_target_key in modules:
                        user_module = modules.get(user_target_key)
                        if isinstance(user_module, self.qconfig.activation):
                            is_already_observed_by_user = True
                            # 跳过已被观察的输出是 DEBUG 信息
                            self.logger.debug(
                                _colorize(
                                    f"  [跳过][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 的输出 ({reason_for_quantization}) 已被后续用户 "
                                    f"'{_colorize(user.name, COLOR_NODE)}' ({_colorize(user_target_key, COLOR_MODULE)}) 观察。",
                                    COLOR_DEBUG,
                                )
                            )
                            break

                if not is_already_observed_by_user:
                    act_quant_module_cls = self.qconfig.activation
                    act_quant_instance = act_quant_module_cls()
                    quant_module_name = self._get_unique_module_name(output_prefix)
                    traced_model.add_module(quant_module_name, act_quant_instance)

                    # 插入消息是 DEBUG 级别
                    self.logger.debug(
                        _colorize(
                            f"  [{_colorize('插入', COLOR_ACTION)}][{_colorize('激活', COLOR_TARGET)}] 节点 '{_colorize(node.name, COLOR_NODE)}' 需要输出观察。 "
                            f"{_colorize('原因:', COLOR_REASON)} {reason_for_quantization}. "
                            f"插入量化器 '{_colorize(quant_module_name, COLOR_MODULE)}'。",
                            COLOR_DEBUG,
                        )
                    )

                    with graph.inserting_after(node):
                        inserted_node = graph.call_module(
                            quant_module_name, args=(node,)
                        )

                    # 获取当前节点的所有用户
                    users_to_update = list(node.users.keys())
                    updated_users_count = 0
                    # 遍历所有用户并更新它们的输入
                    for user_node in users_to_update:
                        # 确保不更新刚刚插入的量化器节点自身
                        if user_node != inserted_node:
                            user_node.replace_input_with(node, inserted_node)
                            updated_users_count += 1

                    # 确保插入的节点明确以原节点为参数
                    inserted_node.args = (node,)

                    # 插入的细节是 DEBUG 级别
                    self.logger.debug(
                        _colorize(
                            f"    -> 插入 {_colorize('call_module', COLOR_MODULE)} (激活 量化器): '{_colorize(inserted_node.name, COLOR_NODE)}'",
                            COLOR_DEBUG,
                        )
                    )
                    self.logger.debug(
                        _colorize(
                            f"    -> 更新了 '{_colorize(node.name, COLOR_NODE)}' 的 {updated_users_count} 个用户，使其使用 '{_colorize(inserted_node.name, COLOR_NODE)}'。",
                            COLOR_DEBUG,
                        )
                    )

            # 记录跳过激活的原因 *仅当* 处于调试模式 *且* 不是因为它已经是观察器时
            elif (
                self.debug
                and not should_quantize_output
                and not is_output_from_quant_module
                and node.op != "output"  # 输出节点通常不需要观察
            ):
                if not self.qconfig.activation:
                    if not self._printed_no_activation_config:
                        # 将缺少配置记录为 INFO 一次（在非调试模式下也可见）
                        self.logger.info(
                            _colorize(
                                f"  [信息][{_colorize('激活', COLOR_TARGET)}] 跳过激活量化器。 "
                                f"{_colorize('原因:', COLOR_REASON)} QConfig 没有配置激活量化器 (`qconfig.activation` 为 None)。",
                                COLOR_INFO,
                            )
                        )
                        self._printed_no_activation_config = True
                # 其他跳过（不可量化类型等）如果调试开启，已在上面记录。

        # --- 清理和重新编译 ---
        try:
            self.logger.debug(
                _colorize(
                    "\n--> [步骤 3] 校验图 (检查有效性)...",
                    COLOR_INFO,
                )
            )
            graph.lint()  # 检查图的结构是否有效
            prepared_model = GraphModule(traced_model, graph)  # 重新创建 GraphModule
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
                    "\n--- 最终 FX 图 (准备阶段后) ---",
                    COLOR_BOLD + COLOR_INFO,
                )
            )
            prepared_model.graph.print_tabular()  # 直接打印以进行调试
            self.logger.debug(
                _colorize("--- (结束准备图) ---\n", COLOR_BOLD + COLOR_INFO)
            )

        # 使用 logger.info 标记阶段完成（在两种模式下都可见）
        self.logger.info(
            _colorize(
                f"{'='*20} 量化准备阶段完成 {'='*20}",
                COLOR_PHASE + COLOR_SUCCESS,
            )
        )
        self.logger.info(
            _colorize(
                "模型准备完毕：已为权重和激活插入量化器, 量化图构建完毕",
                COLOR_SUCCESS,
            )
        )
        return prepared_model
