# micronet/compression/fx/quantization/tests/core/test_fx_quantizer.py

import pytest
import sys
import torch
import torch.nn as nn
from torch.fx import GraphModule

try:
    from micronet.compression.fx.quantization.core.quantizer import Quantizer
    from micronet.compression.fx.quantization.core.fake_quant import FakeQuantize
    from micronet.compression.fx.quantization.core.qconfig import (
        QConfig,
        default_ptq_qconfig,
        default_qat_qconfig,
    )
    from micronet.compression.fx.quantization.core.graph_utils import (
        _colorize,
        COLOR_SUCCESS,
        COLOR_NODE,
        COLOR_OPERATOR,
        COLOR_TARGET,
    )
except ImportError as e:
    print(
        f"错误：无法导入必要的模块。请确保 quantizer.py, qconfig.py, graph_utils.py, observer.py, fake_quant.py 在 Python 路径中。错误信息: {e}"
    )
    sys.exit(1)


# --- 测试辅助函数 ---
def count_quantizer_nodes(graph_module: GraphModule, qconfig: QConfig):
    """统计图中插入的量化器节点数量 (权重和激活)"""
    weight_quant_count = 0
    act_quant_count = 0
    modules = dict(graph_module.named_modules())

    # 迭代节点来计数，更依赖图结构而非模块类型字符串
    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            module = modules.get(str(node.target))
            if isinstance(module, FakeQuantize):
                # 检查此 FakeQuantize 节点的输入是 get_attr (权重) 还是其他 (激活)
                if node.args and node.args[0].op == "get_attr":
                    if qconfig.weight:  # 只有当 qconfig 配置了权重量化时才计数
                        weight_quant_count += 1
                else:
                    if qconfig.activation:  # 只有当 qconfig 配置了激活量化时才计数
                        # 排除权重量化器（即使它的输入不是get_attr，例如在某些复杂模式下），通过检查它的名字
                        # （这依赖于 Quantizer 的命名约定）
                        # 一个更可靠的方法是在 Quantizer 中为 FakeQuantize 添加一个标志
                        if "weight_fake_quant" not in node.target:
                            act_quant_count += 1

    # 如果上面的逻辑不准确，采用依赖FakeQuantize实例和qconfig
    if weight_quant_count == 0 and act_quant_count == 0:  # 简单回退检查
        weight_quant_count = 0
        act_quant_count = 0
        for node in graph_module.graph.nodes:
            if node.op == "call_module":
                module = modules.get(str(node.target))
                if module:
                    if (
                        qconfig.weight
                        and isinstance(module, FakeQuantize)
                        and "weight" in node.target
                    ):
                        weight_quant_count += 1
                    elif (
                        qconfig.activation
                        and isinstance(module, FakeQuantize)
                        and "act" in node.target
                    ):
                        act_quant_count += 1

    return weight_quant_count, act_quant_count


def get_node_by_name(graph: torch.fx.Graph, name: str):
    """通过名称查找节点"""
    for node in graph.nodes:
        if node.name == name:
            return node
    return None


def print_node_info(node: torch.fx.Node, prefix="  "):
    """打印节点的简要信息"""
    users = [n.name for n in node.users]
    args_repr = []
    for a in node.args:
        if isinstance(a, torch.fx.Node):
            args_repr.append(a.name)
        elif isinstance(a, list) or isinstance(a, tuple):
            args_repr.append(
                f"[{','.join(n.name for n in a if isinstance(n, torch.fx.Node))}]"
            )  # 显示列表/元组中的节点名
        else:
            args_repr.append(str(a))
    kwargs_repr = {
        k: (v.name if isinstance(v, torch.fx.Node) else str(v))
        for k, v in node.kwargs.items()
    }

    return f"{prefix}节点: {_colorize(node.name, COLOR_NODE)}, Op: {_colorize(node.op, COLOR_OPERATOR)}, Target: {_colorize(str(node.target), COLOR_TARGET)}, Args: {args_repr}, Kwargs: {kwargs_repr}, Users: {users}"


# --- 定义测试模型 ---


# 1. 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(8 * 28 * 28, 10)  # 假设输入是 3x28x28

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        # 添加一个函数调用和一个方法调用
        y = torch.add(x, 1.0)  # 函数调用
        z = y.mul(2.0)  # 方法调用
        out = self.linear(z)
        return out


# 2. VGG 风格块 (简化)
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        return x


# 3. ResNet 风格块 (BasicBlock)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接 (add 操作)
        out += identity  # 等价于 torch.add(out, identity) 或 out.add(identity)
        out = self.relu(out)

        return out


# 4. MobileNetV2 风格块 (InvertedResidual - 简化)
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend(
            [
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # add 操作
        else:
            return self.conv(x)


# --- 测试类 ---
class TestQuantizer:

    # 使用 fixture 提供 QConfig
    @pytest.fixture
    def ptq_qconfig(self):
        return default_ptq_qconfig

    @pytest.fixture
    def qat_qconfig(self):
        return default_qat_qconfig

    def run_prepare_test(
        self, model_name, model, qconfig, expected_weights, expected_acts, debug=False
    ):
        """运行 prepare 测试并进行基本断言的辅助函数"""
        print(
            f"\n--- 测试: {model_name} (Prepare, QConfig: {qconfig}, Debug: {debug}) ---"
        )
        quantizer = Quantizer(
            qconfig=qconfig, fuse_bn=False, debug=debug
        )  # 显式禁用 BN 融合，除非专门测试
        prepared_model = quantizer.prepare(model.eval())  # 确保是 eval 模式

        assert isinstance(prepared_model, GraphModule)

        # 检查插入的量化器数量
        w_count, a_count = count_quantizer_nodes(prepared_model, qconfig)
        print(f"  预期权重量化器: {expected_weights}, 找到: {w_count}")
        print(f"  预期激活量化器: {expected_acts}, 找到: {a_count}")

        assert w_count == expected_weights, f"{model_name}: 权重量化器数量不匹配"
        assert a_count == expected_acts, f"{model_name}: 激活量化器数量不匹配"

        # 详细检查图结构 (如果 debug 开启)
        if debug:
            print("  图节点概览 (部分):")
            modules = dict(prepared_model.named_modules())
            limit = 25  # 增加显示限制
            count = 0
            for node in prepared_model.graph.nodes:
                if (
                    count < limit or "fake_quant" in node.name
                ):  # 显示前几个或包含 fake_quant 的
                    print(print_node_info(node))
                elif count == limit:
                    print("  ...")
                count += 1

                # --- 检查权重量化器插入 ---
                is_weight_fq = (
                    qconfig.weight
                    and node.op == "call_module"
                    and "weight_fake_quant" in node.target
                )
                if is_weight_fq:
                    # print(f"    检查权重量化器: {node.name}")
                    assert len(node.args) == 1, f"权重量化器 {node.name} 应有 1 个参数"
                    input_node = node.args[0]
                    assert (
                        input_node.op == "get_attr"
                    ), f"权重量化器 {node.name} 的输入应为 get_attr, 实际为 {input_node.op}"
                    # 检查原始模块是否使用了这个量化后的权重
                    found_user = False
                    original_module_node = None
                    for user_node in node.users:
                        # 权重 FQ 的用户应该是原始模块 (Conv/Linear)
                        if user_node.op == "call_module" and not isinstance(
                            modules.get(str(user_node.target)), FakeQuantize
                        ):
                            original_module_node = user_node
                            # 检查原始模块的参数是否包含这个 weight fq 节点
                            if any(
                                arg == node for arg in original_module_node.args
                            ) or any(
                                kwarg == node
                                for kwarg in original_module_node.kwargs.values()
                            ):
                                found_user = True
                                break
                            # 特殊情况：对于 F.conv2d 等函数调用
                            elif (
                                original_module_node.op == "call_function"
                                and len(original_module_node.args) > 1
                                and original_module_node.args[1] == node
                            ):
                                found_user = True
                                break

                    assert (
                        found_user
                    ), f"原始模块 {original_module_node.name if original_module_node else '未找到'} 应使用权重量化器 {node.name}"

                # --- 检查激活量化器插入 ---
                is_act_fq = (
                    qconfig.activation
                    and node.op == "call_module"
                    and "act_fake_quant" in node.target
                )
                if is_act_fq:
                    # print(f"    检查激活量化器: {node.name}")
                    assert len(node.args) == 1, f"激活量化器 {node.name} 应有 1 个参数"
                    input_node = node.args[0]  # 获取被观察的节点

                    assert (
                        node in input_node.users
                    ), f"激活量化器 {node.name} 应是其输入 {input_node.name} 的用户"

                    # 检查原始节点的所有 *其他* 用户是否现在都使用了这个量化器
                    original_users = list(input_node.users)
                    for user_node in original_users:
                        if user_node != node:  # 跳过量化器本身
                            # 检查 user_node 的参数是否已从 input_node 替换为 node
                            new_args = list(user_node.args)
                            new_kwargs = dict(user_node.kwargs)
                            replaced = False
                            for i, arg in enumerate(new_args):
                                if arg == input_node:
                                    # 理论上 Quantizer 应该已经替换了，所以这里断言 arg 应该是 node
                                    assert (
                                        arg == node
                                    ), f"节点 {user_node.name} 的参数 {i} 应为 {node.name} 而不是 {input_node.name}"
                                    replaced = (
                                        True  # 虽然断言失败会停止，但逻辑上是替换了
                                    )
                            for k, v in new_kwargs.items():
                                if v == input_node:
                                    assert (
                                        v == node
                                    ), f"节点 {user_node.name} 的关键字参数 '{k}' 应为 {node.name} 而不是 {input_node.name}"
                                    replaced = True

                            # 这个断言可能过于严格，因为可能有些用户确实不再需要这个输入了
                            # 但对于标准流程，原始用户应该被重定向到使用量化器输出
                            assert (
                                replaced
                            ), f"节点 {user_node.name} 应该使用量化器 {node.name} 而不是 {input_node.name}"

        print(_colorize(f"--- 测试通过: {model_name} (Prepare) ---", COLOR_SUCCESS))
        return prepared_model

    # --- 测试用例 (注入 fixtures) ---

    def test_01_init_valid(self, ptq_qconfig, qat_qconfig):  # 注入
        """测试 Quantizer 初始化 (有效 QConfig)"""
        print("\n--- 测试: Quantizer 初始化 (有效) ---")
        quantizer_ptq = Quantizer(qconfig=ptq_qconfig)
        assert isinstance(quantizer_ptq, Quantizer)
        quantizer_qat = Quantizer(qconfig=qat_qconfig)
        assert isinstance(quantizer_qat, Quantizer)
        print(_colorize("--- 测试通过: Quantizer 初始化 (有效) ---", COLOR_SUCCESS))

    def test_02_init_invalid_qconfig_type(self):
        """测试 Quantizer 初始化 (无效 QConfig 类型)"""
        print("\n--- 测试: Quantizer 初始化 (无效 QConfig 类型) ---")
        with pytest.raises(
            ValueError, match="qconfig 必须是 QConfig 的实例"
        ):  # 使用 pytest.raises
            Quantizer(qconfig="not a qconfig")
        print(
            _colorize(
                "--- 测试通过: Quantizer 初始化 (无效 QConfig 类型) ---", COLOR_SUCCESS
            )
        )

    def test_03_init_invalid_qconfig_callable(self):
        """测试 Quantizer 初始化 (无效 QConfig 可调用对象)"""
        print("\n--- 测试: Quantizer 初始化 (无效 QConfig 可调用对象) ---")
        invalid_cfg1 = QConfig(activation="not callable", weight=FakeQuantize)
        with pytest.raises(
            TypeError, match="qconfig.activation 必须是可调用的工厂函数或 None"
        ):  # 使用 pytest.raises
            Quantizer(qconfig=invalid_cfg1)

        invalid_cfg2 = QConfig(activation=FakeQuantize, weight=123)
        with pytest.raises(
            TypeError, match="qconfig.weight 必须是可调用的工厂函数或 None"
        ):  # 使用 pytest.raises
            Quantizer(qconfig=invalid_cfg2)
        print(
            _colorize(
                "--- 测试通过: Quantizer 初始化 (无效 QConfig 可调用对象) ---",
                COLOR_SUCCESS,
            )
        )

    def test_04_prepare_simple_model_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: SimpleModel (PTQ)"""
        model = SimpleModel()
        expected_weights = 2  # conv, linear
        expected_acts = 6  # input, conv, relu, add, mul, linear
        prepared = self.run_prepare_test(
            "SimpleModel_PTQ",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,  # 改为 False 减少默认输出，需要时手动改为 True
        )

    def test_05_prepare_simple_model_qat(self, qat_qconfig):  # 注入
        """测试 prepare: SimpleModel (QAT FakeQuant)"""
        model = SimpleModel()
        expected_weights = 2
        expected_acts = 6
        prepared = self.run_prepare_test(
            "SimpleModel_QAT",
            model,
            qat_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_06_prepare_simple_model_no_activation(self):  # 不需要 qconfig fixture
        """测试 prepare: SimpleModel (仅权重量化)"""
        model = SimpleModel()
        qconfig = QConfig(activation=None, weight=FakeQuantize)  # 直接定义
        expected_weights = 2
        expected_acts = 0
        prepared = self.run_prepare_test(
            "SimpleModel_WeightOnly",
            model,
            qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_07_prepare_simple_model_no_weight(self):  # 不需要 qconfig fixture
        """测试 prepare: SimpleModel (仅激活量化)"""
        model = SimpleModel()
        qconfig = QConfig(activation=FakeQuantize, weight=None)  # 直接定义
        expected_weights = 0
        expected_acts = 6
        prepared = self.run_prepare_test(
            "SimpleModel_ActOnly",
            model,
            qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_08_prepare_vgg_block_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: VGGBlock (PTQ)"""
        model = VGGBlock(in_channels=16, out_channels=32)
        expected_weights = 2  # conv1, conv2
        expected_acts = 6  # input, conv1, relu1, conv2, relu2, pool
        prepared = self.run_prepare_test(
            "VGGBlock_PTQ",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    # 目前以下测试用例的BN单独量化处理, 后续会添加量化Pattern
    def test_09_prepare_resnet_block_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: BasicBlock (ResNet, PTQ, fuse_bn=False)"""
        model = BasicBlock(inplanes=16, planes=16)
        # 预期 (不融合 BN):
        # 权重: conv1, bn1, conv2, bn2 = 4
        # 激活: input, conv1, bn1, relu, conv2, bn2, add, final_relu = 8
        expected_weights = 4
        expected_acts = 8
        prepared = self.run_prepare_test(
            "ResNetBlock_PTQ_NoFuse",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_10_prepare_resnet_block_downsample_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: BasicBlock (ResNet with Downsample, PTQ, fuse_bn=False)"""
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        model = BasicBlock(inplanes=16, planes=32, stride=2, downsample=downsample)
        # 预期 (不融合 BN):
        # 权重: conv1, bn1, conv2, bn2, downsample.0, ds_bn = 6
        # 激活: input, conv1, bn1, relu, conv2, bn2, ds_conv, ds_bn, add, final_relu = 10
        expected_weights = 6
        expected_acts = 10
        prepared = self.run_prepare_test(
            "ResNetBlockDownsample_PTQ_NoFuse",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_11_prepare_mobilenetv2_block_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: InvertedResidual (MobileNetV2, PTQ, fuse_bn=False)"""
        model = InvertedResidual(inp=16, oup=16, stride=1, expand_ratio=6)
        # 预期 (不融合 BN):
        # 权重: conv.0(pw), bn1, conv.3(dw), bn4, conv.6(pwl), bn7 = 6
        # 激活: input, conv.0, bn1, relu2, conv.3, bn4, relu5, conv.6, bn7, add = 10
        expected_weights = 6
        expected_acts = 10
        prepared = self.run_prepare_test(
            "MobileNetV2Block_PTQ_NoFuse",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_12_prepare_mobilenetv2_block_no_residual_ptq(self, ptq_qconfig):  # 注入
        """测试 prepare: InvertedResidual (MobileNetV2, No Residual, PTQ, fuse_bn=False)"""
        model = InvertedResidual(inp=16, oup=24, stride=2, expand_ratio=6)
        # 预期 (不融合 BN, 无 add):
        # 权重: conv.0(pw), bn1, conv.3(dw), bn4, conv.6(pwl), bn7 = 6
        # 激活: input, conv.0, bn1, relu2, conv.3, bn4, relu5, conv.6, bn7 = 9
        expected_weights = 6
        expected_acts = 9
        prepared = self.run_prepare_test(
            "MobileNetV2BlockNoRes_PTQ_NoFuse",
            model,
            ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )
