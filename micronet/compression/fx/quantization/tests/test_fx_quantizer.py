import unittest
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        is_quantizable_weight_module,
        is_quantizable_activation_module,
        is_quantizable_activation_function,
        is_quantizable_activation_method,
        _colorize,  # 导入颜色函数以便测试输出使用
        COLOR_SUCCESS,
        COLOR_ERROR,
        COLOR_BOLD,
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

    for node in graph_module.graph.nodes:
        if node.op == "call_module":
            module = modules.get(str(node.target))
            if module:
                # 检查这个权重量化器的输入是否是 get_attr (通常是)
                if node.args and node.args[0].op == "get_attr":
                    if qconfig.weight and isinstance(module, FakeQuantize):
                        weight_quant_count += 1
                elif qconfig.activation and isinstance(module, FakeQuantize):
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
    args = [a.name if isinstance(a, torch.fx.Node) else str(a) for a in node.args]
    return f"{prefix}节点: {_colorize(node.name, COLOR_NODE)}, Op: {_colorize(node.op, COLOR_OPERATOR)}, Target: {_colorize(str(node.target), COLOR_TARGET)}, Args: {args}, Users: {users}"


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
        self.pool = nn.MaxPool2d(2, 2)  # MaxPool 输出通常不量化，但可观察

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


class TestQuantizer(unittest.TestCase):

    def setUp(self):
        """在每个测试方法运行前设置"""
        # 默认 QConfig
        self.ptq_qconfig = default_ptq_qconfig
        self.qat_qconfig = default_qat_qconfig

    def run_prepare_test(
        self, model_name, model, qconfig, expected_weights, expected_acts, debug=False
    ):
        """运行 prepare 测试并进行基本断言的辅助函数"""
        print(
            f"\n--- 测试: {model_name} (Prepare, QConfig: {qconfig}, Debug: {debug}) ---"
        )
        quantizer = Quantizer(qconfig=qconfig, debug=debug)
        prepared_model = quantizer.prepare(model.eval())  # 确保是 eval 模式

        self.assertIsInstance(prepared_model, GraphModule)

        # 检查插入的量化器数量
        w_count, a_count = count_quantizer_nodes(prepared_model, qconfig)
        print(f"  预期权重量化器: {expected_weights}, 找到: {w_count}")
        print(f"  预期激活量化器: {expected_acts}, 找到: {a_count}")

        self.assertEqual(
            w_count, expected_weights, f"{model_name}: 权重量化器数量不匹配"
        )
        self.assertEqual(a_count, expected_acts, f"{model_name}: 激活量化器数量不匹配")

        # 详细检查图结构
        # TODO: 打开 debug 时的详细检查
        if debug:
            print("  图节点概览 (部分):")
            modules = dict(prepared_model.named_modules())
            limit = 15
            count = 0
            for node in prepared_model.graph.nodes:
                if count < limit:  # or 'obs' in node.name : # 只显示前几个或包含 obs 的
                    print(print_node_info(node))
                elif count == limit:
                    print("  ...")
                count += 1
                # NOTE: 目前weight和activation的量化器一样，这里无法区分
                # TODO: 区分weight和activation的量化器
                # 检查权重量化器是否正确插入在其使用者之前
                if qconfig.weight and node.op == "call_module":
                    module = modules.get(str(node.target))
                    if module and isinstance(module, FakeQuantize):
                        # 检查输入是否 get_attr
                        self.assertEqual(len(node.args), 1)
                        input_node = node.args[0]
                        self.assertEqual(
                            input_node.op,
                            "get_attr",
                            f"权重量化器 {node.name} 的输入应为 get_attr",
                        )
                        # 检查 get_attr 的用户是否包含原始模块
                        original_module_node = list(node.users)[
                            0
                        ]  # 获取使用此量化权重的节点
                        self.assertTrue(
                            any(arg == node for arg in original_module_node.args)
                            or node in original_module_node.kwargs.values(),
                            f"原始模块 {original_module_node.name} 应使用权重量化器 {node.name}",
                        )

                # 检查激活量化器是否在其源节点之后插入
                if qconfig.activation and node.op == "call_module":
                    module = modules.get(str(node.target))
                    if module and isinstance(module, FakeQuantize):
                        self.assertEqual(len(node.args), 1)
                        input_node = node.args[0]  # 获取被观察的节点
                        self.assertIn(
                            node,
                            input_node.users,
                            f"激活量化器 {node.name} 应是其输入 {input_node.name} 的用户",
                        )
                        # 检查原始节点的用户是否被更新为使用量化器
                        original_users = list(input_node.users)
                        self.assertIn(
                            node,
                            original_users,
                            f"激活量化器 {node.name} 应该是 {input_node.name} 的用户",
                        )
                        for user in original_users:
                            if user != node:  # 跳过量化器本身
                                self.assertTrue(
                                    any(arg == node for arg in user.args)
                                    or node in user.kwargs.values(),
                                    f"节点 {user.name} 应该使用量化器 {node.name} 而不是 {input_node.name}",
                                )

        print(_colorize(f"--- 测试通过: {model_name} (Prepare) ---", COLOR_SUCCESS))
        return prepared_model

    # --- 测试用例 ---

    def test_01_init_valid(self):
        """测试 Quantizer 初始化 (有效 QConfig)"""
        print("\n--- 测试: Quantizer 初始化 (有效) ---")
        quantizer_ptq = Quantizer(qconfig=self.ptq_qconfig)
        self.assertIsInstance(quantizer_ptq, Quantizer)
        quantizer_qat = Quantizer(qconfig=self.qat_qconfig)
        self.assertIsInstance(quantizer_qat, Quantizer)
        print(_colorize("--- 测试通过: Quantizer 初始化 (有效) ---", COLOR_SUCCESS))

    def test_02_init_invalid_qconfig_type(self):
        """测试 Quantizer 初始化 (无效 QConfig 类型)"""
        print("\n--- 测试: Quantizer 初始化 (无效 QConfig 类型) ---")
        with self.assertRaisesRegex(ValueError, "qconfig 必须是 QConfig 的实例"):
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
        with self.assertRaisesRegex(
            TypeError, "qconfig.activation 必须是可调用的工厂函数或 None"
        ):
            Quantizer(qconfig=invalid_cfg1)

        invalid_cfg2 = QConfig(activation=FakeQuantize, weight=123)
        with self.assertRaisesRegex(
            TypeError, "qconfig.weight 必须是可调用的工厂函数或 None"
        ):
            Quantizer(qconfig=invalid_cfg2)
        print(
            _colorize(
                "--- 测试通过: Quantizer 初始化 (无效 QConfig 可调用对象) ---",
                COLOR_SUCCESS,
            )
        )

    def test_04_prepare_simple_model_ptq(self):
        """测试 prepare: SimpleModel (PTQ)"""
        model = SimpleModel()
        # 预期:
        # - conv.weight: 1
        # - linear.weight: 1
        # 激活:
        # - 输入 x: 1
        # - conv 输出: 1
        # - relu 输出: 1
        # - add 输出: 1
        # - mul 输出: 1
        # - linear 输出: 1 (通常会观察线性层输出)
        expected_weights = 2
        expected_acts = 6  # 输入, conv, relu, add, mul, linear
        prepared = self.run_prepare_test(
            "SimpleModel_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_05_prepare_simple_model_qat(self):
        """测试 prepare: SimpleModel (QAT FakeQuant)"""
        model = SimpleModel()
        # 预期数量与 PTQ 相同，但类型是 PlaceholderFakeQuant
        expected_weights = 2
        expected_acts = 6
        prepared = self.run_prepare_test(
            "SimpleModel_QAT",
            model,
            self.qat_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )  # debug=False 减少输出

    def test_06_prepare_simple_model_no_activation(self):
        """测试 prepare: SimpleModel (仅权重量化)"""
        model = SimpleModel()
        qconfig = QConfig(activation=None, weight=FakeQuantize)
        expected_weights = 2
        expected_acts = 0  # 不量化激活
        prepared = self.run_prepare_test(
            "SimpleModel_WeightOnly",
            model,
            qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_07_prepare_simple_model_no_weight(self):
        """测试 prepare: SimpleModel (仅激活量化)"""
        model = SimpleModel()
        qconfig = QConfig(activation=FakeQuantize, weight=None)
        expected_weights = 0  # 不量化权重
        expected_acts = 6
        prepared = self.run_prepare_test(
            "SimpleModel_ActOnly",
            model,
            qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_08_prepare_vgg_block_ptq(self):
        """测试 prepare: VGGBlock (PTQ)"""
        model = VGGBlock(in_channels=16, out_channels=32)
        # 预期:
        # - conv1.weight: 1
        # - conv2.weight: 1
        # 激活:
        # - 输入 x: 1
        # - conv1 输出: 1
        # - relu1 输出: 1
        # - conv2 输出: 1
        # - relu2 输出: 1
        # - pool 输出: 1 (MaxPool2d 在列表中)
        expected_weights = 2
        expected_acts = 6
        prepared = self.run_prepare_test(
            "VGGBlock_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_09_prepare_resnet_block_ptq(self):
        """测试 prepare: BasicBlock (ResNet, PTQ)"""
        # 无下采样
        model = BasicBlock(inplanes=16, planes=16)
        # 预期:
        # - conv1.weight: 1
        # - conv2.weight: 1
        # 激活:
        # - 输入 x: 1
        # - conv1 输出: 1
        # - bn1 输出: 1
        # - relu (after bn1) 输出: 1
        # - conv2 输出: 1
        # - bn2 输出: 1
        # - add (残差连接) 输出: 1
        # - relu (final) 输出: 1
        expected_weights = 4
        expected_acts = 8
        prepared = self.run_prepare_test(
            "ResNetBlock_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_10_prepare_resnet_block_downsample_ptq(self):
        """测试 prepare: BasicBlock (ResNet with Downsample, PTQ)"""
        # 带下采样 (通常是 Conv+BN)
        downsample = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32),
        )
        model = BasicBlock(inplanes=16, planes=32, stride=2, downsample=downsample)
        # 预期:
        # - conv1.weight: 1
        # - conv2.weight: 1
        # - downsample.0.weight (Conv): 1
        # 激活:
        # - 输入 x: 1
        # - conv1 输出: 1
        # - bn1 输出: 1
        # - relu (after bn1) 输出: 1
        # - conv2 输出: 1
        # - bn2 输出: 1
        # - downsample.0 (Conv) 输出: 1
        # - downsample.1 (BN) 输出: 1
        # - add (残差连接) 输出: 1
        # - relu (final) 输出: 1
        expected_weights = 6
        expected_acts = 10
        prepared = self.run_prepare_test(
            "ResNetBlockDownsample_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_11_prepare_mobilenetv2_block_ptq(self):
        """测试 prepare: InvertedResidual (MobileNetV2, PTQ)"""
        # expand_ratio=6, stride=1, inp=16, oup=16 (带残差连接)
        model = InvertedResidual(inp=16, oup=16, stride=1, expand_ratio=6)
        hidden_dim = 16 * 6
        # 预期:
        # - conv.0 (pw Conv): 1
        # - conv.3 (dw Conv): 1
        # - conv.6 (pw-linear Conv): 1
        # 激活:
        # - 输入 x: 1
        # - conv.0 (pw Conv) 输出: 1
        # - conv.1 (BN) 输出: 1
        # - conv.2 (ReLU6) 输出: 1
        # - conv.3 (dw Conv) 输出: 1
        # - conv.4 (BN) 输出: 1
        # - conv.5 (ReLU6) 输出: 1
        # - conv.6 (pw-linear Conv) 输出: 1
        # - conv.7 (BN) 输出: 1
        # - add (残差) 输出: 1
        expected_weights = 6
        expected_acts = 10
        prepared = self.run_prepare_test(
            "MobileNetV2Block_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_12_prepare_mobilenetv2_block_no_residual_ptq(self):
        """测试 prepare: InvertedResidual (MobileNetV2, No Residual, PTQ)"""
        # stride=2, inp=16, oup=24 (无残差连接)
        model = InvertedResidual(inp=16, oup=24, stride=2, expand_ratio=6)
        hidden_dim = 16 * 6
        # 预期 (同上，但最后没有 add)
        expected_weights = 6
        expected_acts = 9  # 输入 + 8 层 conv sequential 的输出
        prepared = self.run_prepare_test(
            "MobileNetV2BlockNoRes_PTQ",
            model,
            self.ptq_qconfig,
            expected_weights,
            expected_acts,
            debug=False,
        )

    def test_13_graph_utils_coverage(self):
        """测试 graph_utils 中的辅助函数"""
        print("\n--- 测试: graph_utils 辅助函数 ---")
        # 权重可量化
        self.assertTrue(is_quantizable_weight_module(nn.Conv2d(3, 3, 3)))
        self.assertTrue(is_quantizable_weight_module(nn.Linear(10, 5)))
        self.assertTrue(is_quantizable_weight_module(nn.Embedding(100, 10)))
        self.assertTrue(is_quantizable_weight_module(nn.BatchNorm2d(3)))
        self.assertFalse(is_quantizable_weight_module(nn.ReLU()))

        # 激活可量化 (模块)
        self.assertTrue(is_quantizable_activation_module(nn.Conv2d(3, 3, 3)))
        self.assertTrue(is_quantizable_activation_module(nn.Linear(10, 5)))
        self.assertTrue(is_quantizable_activation_module(nn.BatchNorm2d(3)))
        self.assertTrue(is_quantizable_activation_module(nn.ReLU()))
        self.assertTrue(is_quantizable_activation_module(nn.MaxPool2d(2)))
        self.assertTrue(is_quantizable_activation_module(nn.AdaptiveAvgPool2d(1)))
        self.assertFalse(
            is_quantizable_activation_module(nn.Dropout())
        )  # Dropout 通常不量化

        # 激活可量化 (函数)
        self.assertTrue(is_quantizable_activation_function(torch.add))
        self.assertTrue(is_quantizable_activation_function(F.relu))
        self.assertTrue(is_quantizable_activation_function(torch.cat))
        self.assertTrue(is_quantizable_activation_function(F.adaptive_avg_pool2d))
        self.assertFalse(
            is_quantizable_activation_function(torch.randn)
        )  # 不是计算函数
        self.assertFalse(is_quantizable_activation_function(F.dropout))

        # 激活可量化 (方法)
        self.assertTrue(is_quantizable_activation_method("add"))
        self.assertTrue(is_quantizable_activation_method("__add__"))
        self.assertTrue(is_quantizable_activation_method("relu_"))
        self.assertTrue(is_quantizable_activation_method("mean"))
        self.assertFalse(is_quantizable_activation_method("view"))
        self.assertFalse(is_quantizable_activation_method("transpose"))

        print(_colorize("--- 测试通过: graph_utils 辅助函数 ---", COLOR_SUCCESS))


# --- 运行测试 ---
if __name__ == "__main__":
    # 使用 unittest TestLoader 来查找和运行测试
    suite = unittest.TestSuite()

    # 按顺序添加测试，以便输出更连贯
    suite.addTest(unittest.makeSuite(TestQuantizer))

    # 创建一个 TextTestRunner，设置 verbosity=2 以获取更详细的输出
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)  # 输出到标准输出

    # 运行测试套件
    print("=" * 70)
    print("开始运行 Quantizer 测试套件...")
    print("=" * 70)
    result = runner.run(suite)
    print("=" * 70)
    print("测试运行结束。")
    print("=" * 70)

    # 根据结果退出，方便 CI/CD 集成
    if result.wasSuccessful():
        print(_colorize("所有测试通过！", COLOR_BOLD + COLOR_SUCCESS))
        sys.exit(0)
    else:
        print(_colorize("存在测试失败！", COLOR_BOLD + COLOR_ERROR))
        sys.exit(1)
