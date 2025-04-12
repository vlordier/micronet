# quantization_framework/quant_core/quantizer.py
import torch.nn as nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule
import sys  # 用于检查 TTY

# from .qconfig import QConfig # Keep if QConfig is defined locally
# 假设这些在 qconfig.py 中定义
from .qconfig import QConfig

from .graph_utils import (
    is_quantizable_weight_module,
    is_quantizable_activation_module,
    is_quantizable_activation_function,
)

# --- ANSI 颜色代码 ---
# 检查是否在 TTY 环境中，以避免在不支持颜色的地方（如文件重定向）输出颜色代码
_IS_TTY = sys.stdout.isatty()

COLOR_DEBUG = "\033[96m"  # 青色 (Cyan) for debug steps
COLOR_INFO = "\033[94m"  # 蓝色 (Blue) for general info
COLOR_WARN = "\033[93m"  # 黄色 (Yellow) for warnings
COLOR_SUCCESS = "\033[92m"  # 绿色 (Green) for success
COLOR_ERROR = "\033[91m"  # 红色 (Red) for errors
COLOR_BOLD = "\033[1m"
COLOR_RESET = "\033[0m"  # 重置颜色


def _colorize(text: str, color_code: str) -> str:
    """如果环境支持，则为文本添加颜色"""
    return f"{color_code}{text}{COLOR_RESET}" if _IS_TTY else text


class Quantizer:
    def __init__(self, qconfig: QConfig, debug: bool = False):
        if not isinstance(qconfig, QConfig):
            raise ValueError("qconfig must be an instance of QConfig")
        # 在初始化时就检查工厂函数是否可调用 (如果不是 None)
        if qconfig.activation is not None and not callable(qconfig.activation):
            raise TypeError(
                f"qconfig.activation must be a callable factory or None, but got {type(qconfig.activation)}"
            )
        if qconfig.weight is not None and not callable(qconfig.weight):
            raise TypeError(
                f"qconfig.weight must be a callable factory or None, but got {type(qconfig.weight)}"
            )

        self.qconfig = qconfig
        self.debug = debug  # 存储 debug 标志
        self._insertion_point_counter = 0
        if self.debug:
            print(_colorize("Quantizer initialized in DEBUG mode.", COLOR_INFO))

    def _get_unique_module_name(self, prefix: str) -> str:
        """生成唯一的模块名称以避免冲突"""
        name = f"{prefix}_{self._insertion_point_counter}"
        self._insertion_point_counter += 1
        return name

    def prepare(self, model: nn.Module) -> GraphModule:
        """
        准备模型以进行量化（PTQ 校准或 QAT 训练）。
        插入激活和权重的占位符（通常是 Observer 或 FakeQuant）。
        动态插入权重的 get_attr 和观察者调用。
        """
        if self.debug:
            print(
                _colorize(
                    "\n--- Starting Model Preparation ---", COLOR_BOLD + COLOR_INFO
                )
            )

        # --- 模型追踪 ---
        try:
            original_training_state = model.training
            model.eval()  # 追踪前设置为 eval 模式更稳定
            tracer = fx.Tracer()
            graph = tracer.trace(model)
            # 创建 GraphModule 时使用原始模型，以保留模块引用
            traced_model = GraphModule(model, graph)
            if self.debug:
                print(_colorize("模型成功追踪！", COLOR_SUCCESS))
            # --- DEBUG: 打印初始图 ---
            if self.debug:
                print(_colorize("\n--- 初始计算图 ---", COLOR_INFO))
                graph.print_tabular()
                print(_colorize("------------------\n", COLOR_INFO))
            # 恢复模型原始的训练状态
            model.train(original_training_state)
        except Exception as e:
            print(_colorize(f"模型追踪失败: {e}", COLOR_ERROR))
            raise e

        graph = traced_model.graph
        # 使用 GraphModule 的 named_modules 获取正确的模块字典
        modules = dict(traced_model.named_modules())
        self._insertion_point_counter = 0  # Reset counter for this preparation

        nodes_to_process = list(
            graph.nodes
        )  # Create a copy to iterate over as we modify the graph
        if self.debug:
            print(
                _colorize(
                    f"开始处理 {len(nodes_to_process)} 个计算图节点...", COLOR_INFO
                )
            )

        for node in nodes_to_process:
            if self.debug:
                print(
                    _colorize(
                        f"\nProcessing node: {node.name} (op={node.op}, target={node.target})",
                        COLOR_DEBUG,
                    )
                )

            # --- 动态插入权重占位符 ---
            if node.op == "call_module":
                if node.target not in modules:
                    if self.debug:
                        print(
                            _colorize(
                                f"  [跳过权重] 节点 '{node.name}' 的目标 '{node.target}' 不在模块字典中。",
                                COLOR_WARN,
                            )
                        )
                    continue

                target_module = modules.get(node.target)

                if (
                    target_module
                    and is_quantizable_weight_module(target_module)
                    and self.qconfig.weight
                    and hasattr(target_module, "weight")
                ):
                    weight_quant_module_cls = self.qconfig.weight
                    weight_quant_instance = weight_quant_module_cls()
                    quant_module_name = self._get_unique_module_name(
                        f"weight_quant_{str(node.target).replace('.', '_')}"
                    )
                    traced_model.add_module(quant_module_name, weight_quant_instance)

                    weight_attr_target = f"{node.target}.weight"
                    if self.debug:
                        print(
                            _colorize(
                                f"  [权重] 发现可量化权重模块: '{node.target}'. "
                                f"准备为权重 '{weight_attr_target}' 插入占位符 '{quant_module_name}'",
                                COLOR_DEBUG,
                            )
                        )

                    with graph.inserting_before(node):
                        get_attr_node = graph.get_attr(weight_attr_target)
                        observer_call_node = graph.call_module(
                            quant_module_name, args=(get_attr_node,)
                        )

                    if self.debug:
                        print(
                            _colorize(
                                f"    插入 get_attr: '{get_attr_node.name}'",
                                COLOR_DEBUG,
                            )
                        )
                        print(
                            _colorize(
                                f"    插入 call_module (权重占位符): '{observer_call_node.name}'",
                                COLOR_DEBUG,
                            )
                        )
                elif (
                    self.debug
                    and target_module
                    and is_quantizable_weight_module(target_module)
                ):
                    if not self.qconfig.weight:
                        print(
                            _colorize(
                                f"  [跳过权重] 模块 '{node.target}' 可量化权重，但 QConfig 未配置权重占位符。",
                                COLOR_INFO,
                            )
                        )
                    elif not hasattr(target_module, "weight"):
                        print(
                            _colorize(
                                f"  [跳过权重] 模块 '{node.target}' 没有 'weight' 属性。",
                                COLOR_WARN,
                            )
                        )

            # --- 插入激活占位符 ---
            should_quantize_output = False
            output_prefix = ""
            reason = ""  # For debug logging
            is_output_from_quant_module = False

            if node.op == "call_module" and node.target in modules:
                current_module = modules.get(node.target)
                # 检查是否是 QConfig 中定义的激活或权重观察者/伪量化器类型
                # 使用 isinstance 检查实例，而不是类本身
                is_act_quant_inst = self.qconfig.activation and isinstance(
                    current_module, self.qconfig.activation
                )
                is_wt_quant_inst = self.qconfig.weight and isinstance(
                    current_module, self.qconfig.weight
                )
                if is_act_quant_inst or is_wt_quant_inst:
                    is_output_from_quant_module = True
                    if self.debug:
                        print(
                            _colorize(
                                f"  [跳过激活] 节点 '{node.name}' 是量化器模块 '{node.target}' 的调用，不在此后插入。",
                                COLOR_INFO,
                            )
                        )

            if (
                not is_output_from_quant_module and self.qconfig.activation
            ):  # Only proceed if activation quant is configured
                if node.op == "call_module":
                    if node.target in modules:
                        target_module = modules.get(node.target)
                        if target_module and is_quantizable_activation_module(
                            target_module
                        ):
                            should_quantize_output = True
                            output_prefix = (
                                f"act_quant_after_{str(node.target).replace('.', '_')}"
                            )
                            reason = f"来自可量化模块 '{node.target}'"
                elif node.op == "call_function":
                    if is_quantizable_activation_function(node.target):
                        should_quantize_output = True
                        func_name = getattr(node.target, "__name__", str(node.target))
                        output_prefix = f"act_quant_after_{func_name}"
                        reason = f"来自可量化函数 '{func_name}'"
                elif node.op == "placeholder":
                    already_quantized_by_user = False
                    for user in node.users:
                        if user.op == "call_module" and user.target in modules:
                            user_module = modules.get(user.target)
                            if self.qconfig.activation and isinstance(
                                user_module, self.qconfig.activation
                            ):
                                already_quantized_by_user = True
                                if self.debug:
                                    print(
                                        _colorize(
                                            f"  [跳过激活] 输入 placeholder '{node.name}' 的输出已被用户节点 '{user.name}' ({user.target}) 量化。",
                                            COLOR_INFO,
                                        )
                                    )
                                break
                    if not already_quantized_by_user:
                        should_quantize_output = True
                        output_prefix = f"act_quant_input_{node.target}"
                        reason = f"来自输入 placeholder '{node.target}'"

            if should_quantize_output and self.qconfig.activation:
                is_already_quantized_by_user = False
                for user in node.users:
                    if user.op == "call_module" and user.target in modules:
                        user_module = modules.get(user.target)
                        if isinstance(user_module, self.qconfig.activation):
                            is_already_quantized_by_user = True
                            if self.debug:
                                print(
                                    _colorize(
                                        f"  [跳过激活] 节点 '{node.name}' ({reason}) 的输出已被后续用户节点 '{user.name}' ({user.target}) 量化。",
                                        COLOR_INFO,
                                    )
                                )
                            break
                if is_already_quantized_by_user:
                    continue

                act_quant_module_cls = self.qconfig.activation
                act_quant_instance = act_quant_module_cls()
                quant_module_name = self._get_unique_module_name(output_prefix)
                traced_model.add_module(quant_module_name, act_quant_instance)

                if self.debug:
                    print(
                        _colorize(
                            f"  [激活] 发现可量化激活点: 节点 '{node.name}' ({reason}). "
                            f"准备插入占位符 '{quant_module_name}'",
                            COLOR_DEBUG,
                        )
                    )

                with graph.inserting_after(node):
                    inserted_node = graph.call_module(quant_module_name, args=(node,))

                node.replace_all_uses_with(inserted_node)
                inserted_node.args = (node,)  # Crucial step

                if self.debug:
                    print(
                        _colorize(
                            f"    插入 call_module (激活占位符): '{inserted_node.name}' 并更新用户",
                            COLOR_DEBUG,
                        )
                    )

            elif (
                self.debug
                and not should_quantize_output
                and not is_output_from_quant_module
            ):
                if not self.qconfig.activation:
                    print(
                        _colorize(
                            f"  [跳过激活] QConfig 未配置激活占位符。", COLOR_INFO
                        )
                    )
                else:
                    # This condition might be too noisy if printed for every non-quantizable node.
                    # Consider printing only if node *could* have been quantized but wasn't for other reasons.
                    pass
                    # print(_colorize(f"  节点 '{node.name}' (op={node.op}) 未被识别为需要激活量化的点。", COLOR_DEBUG))

        # --- 清理和重新编译 ---
        if self.debug:
            print(_colorize("\nLinting graph...", COLOR_INFO))
        graph.lint()  # Check graph validity
        prepared_model = GraphModule(traced_model, graph)  # Recompile

        if self.debug:
            print(_colorize("\n--- 修改后的计算图 (Prepare 阶段) ---", COLOR_INFO))
            prepared_model.graph.print_tabular()
            print(_colorize("------------------------------------\n", COLOR_INFO))

        print(_colorize("模型准备完成，已插入占位符。", COLOR_SUCCESS))
        return prepared_model

    # --- convert 方法 (添加 debug 打印) ---
    def convert(self, prepared_model: GraphModule) -> GraphModule:
        """
        转换准备好的模型，将占位符替换为实际的量化操作或模块 (简化版：仅移除)。
        """
        if self.debug:
            print(
                _colorize(
                    "\n--- Starting Model Conversion (Simplified: Removing Placeholders) ---",
                    COLOR_BOLD + COLOR_INFO,
                )
            )
        else:
            print("\n开始转换模型 (简化版)...")

        graph = prepared_model.graph
        modules = dict(prepared_model.named_modules())

        nodes_to_remove = []
        replacements = {}  # Store nodes to replace {node_to_replace: replacement_node}

        # 收集需要移除的节点和替换关系
        if self.debug:
            print(
                _colorize(
                    f"开始扫描 {len(graph.nodes)} 个节点以进行转换...", COLOR_INFO
                )
            )

        for node in graph.nodes:
            if node.op == "call_module":
                if node.target not in modules:
                    # This shouldn't happen if prepare worked correctly, but good for robustness
                    if self.debug:
                        print(
                            _colorize(
                                f"  [警告] 转换时发现无法找到模块: {node.target} for node {node.name}",
                                COLOR_WARN,
                            )
                        )
                    continue

                target_module = modules.get(node.target)
                is_act_placeholder = self.qconfig.activation and isinstance(
                    target_module, self.qconfig.activation
                )
                is_wt_placeholder = self.qconfig.weight and isinstance(
                    target_module, self.qconfig.weight
                )

                if is_act_placeholder:
                    if node not in replacements:  # Check before adding
                        # Key insight for removal: replace uses of the observer node with its input node
                        replacements[node] = node.args[0]
                    # Mark observer node itself for removal later
                    nodes_to_remove.append(node)
                    if self.debug:
                        print(
                            _colorize(
                                f"  [转换-激活] 标记移除激活占位符: {node.name} ({node.target}). "
                                f"其用户将使用输入: {node.args[0].name}",
                                COLOR_DEBUG,
                            )
                        )

                elif is_wt_placeholder:
                    # Weight observers are side effects, they don't process main data flow.
                    # Remove the observer call itself.
                    nodes_to_remove.append(node)
                    # Also remove the get_attr node *if* it's only used by this observer.
                    if len(node.args) == 1 and node.args[0].op == "get_attr":
                        get_attr_node = node.args[0]
                        # Check if get_attr node has only one user, which is this observer node
                        if (
                            len(get_attr_node.users) == 1
                            and list(get_attr_node.users.keys())[0] == node
                        ):
                            nodes_to_remove.append(get_attr_node)
                            if self.debug:
                                print(
                                    _colorize(
                                        f"  [转换-权重] 标记移除权重占位符: {node.name} ({node.target}) "
                                        f"及其唯一的 get_attr: {get_attr_node.name}",
                                        COLOR_DEBUG,
                                    )
                                )
                        elif self.debug:
                            print(
                                _colorize(
                                    f"  [转换-权重] 标记移除权重占位符: {node.name} ({node.target}). "
                                    f"但其 get_attr '{get_attr_node.name}' 有其他用户 ({list(get_attr_node.users.keys())})，不移除 get_attr。",
                                    COLOR_WARN,
                                )
                            )
                    elif self.debug:
                        print(
                            _colorize(
                                f"  [转换-权重] 标记移除权重占位符: {node.name} ({node.target}). "
                                f"但其参数不符合预期 (单个 get_attr): {node.args}",
                                COLOR_WARN,
                            )
                        )

        # Apply replacements first
        if self.debug and replacements:
            print(_colorize("\n应用节点替换...", COLOR_INFO))
        for old_node, new_node in replacements.items():
            if self.debug:
                print(
                    _colorize(
                        f"  将节点 '{old_node.name}' 的所有用户替换为 '{new_node.name}'",
                        COLOR_DEBUG,
                    )
                )
            old_node.replace_all_uses_with(new_node)

        # Remove marked nodes (in reverse order to handle dependencies)
        if self.debug and nodes_to_remove:
            print(_colorize("\n移除标记的节点...", COLOR_INFO))
        for node in reversed(nodes_to_remove):
            if self.debug:
                print(
                    _colorize(
                        f"  正在移除节点: {node.name} (target: {node.target})",
                        COLOR_DEBUG,
                    )
                )
            graph.erase_node(node)

        # --- 清理和重新编译 ---
        if self.debug:
            print(_colorize("\nLinting graph after conversion...", COLOR_INFO))
        graph.lint()
        converted_model = GraphModule(
            prepared_model, graph
        )  # Recompile from the modified graph

        if self.debug:
            print(_colorize("\n--- 最终计算图 (Convert 阶段) ---", COLOR_INFO))
            converted_model.graph.print_tabular()
            print(_colorize("----------------------------------\n", COLOR_INFO))

        print(_colorize("模型转换完成 (简化版：已移除占位符)。", COLOR_SUCCESS))
        return converted_model
