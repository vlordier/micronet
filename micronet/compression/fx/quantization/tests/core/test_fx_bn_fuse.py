# micronet/compression/fx/quantization/tests/core/test_fx_bn_fuse.py

import unittest
import copy

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.graph_module import GraphModule

from micronet.compression.fx.quantization.core.graph_utils import (
    fuse_conv_linear_bn_fx,
    _get_nested_module,
)


# --- 辅助函数：手动计算融合后的参数 ---
def calculate_fused_params(conv_or_linear, bn):
    """手动计算 Conv/Linear + BN 融合后的权重和偏置"""
    w_orig = conv_or_linear.weight
    b_orig = conv_or_linear.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    gamma = bn.weight if bn.affine else torch.ones_like(running_mean)
    beta = bn.bias if bn.affine else torch.zeros_like(running_mean)

    scale = gamma / torch.sqrt(running_var + eps)

    # 调整 scale 形状
    scale_shape = [1] * w_orig.dim()
    scale_shape[0] = -1
    w_fused = w_orig * scale.reshape(scale_shape)

    if b_orig is not None:
        b_fused = (b_orig - running_mean) * scale + beta
    else:
        b_fused = (0.0 - running_mean) * scale + beta

    return w_fused, b_fused


# --- 测试类 ---
class TestGraphUtilsBnFuse(unittest.TestCase):

    def setUp(self):
        self.default_input_conv = torch.randn(1, 3, 16, 16)
        self.default_input_linear = torch.randn(1, 10)

        # 为模型状态设置随机种子，确保 BN 统计数据可复现
        torch.manual_seed(0)

    def _trace_model(self, model):
        model.eval()  # 必须在 eval 模式下追踪和融合
        traced_graph = fx.Tracer().trace(model)
        return GraphModule(copy.deepcopy(model), traced_graph)  # 使用副本

    def test_fuse_conv2d_bn2d_affine_bias(self):
        """测试融合 Conv2d (有偏置) 和 BatchNorm2d (affine=True)"""
        model_orig = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(8, affine=True),
        )
        # 模拟训练，填充 BN 统计数据
        model_orig.train()
        for _ in range(2):  # 跑几次以获得非零的 running_var
            model_orig(torch.randn(4, 3, 16, 16))
        model_orig.eval()  # 切换回 eval

        # 计算预期结果
        conv_orig = model_orig[0]
        bn_orig = model_orig[1]
        expected_w, expected_b = calculate_fused_params(conv_orig, bn_orig)

        # 追踪和融合
        gm = self._trace_model(model_orig)
        gm_fused = fuse_conv_linear_bn_fx(
            gm, [["0", "1"]]
        )  # 传入 GraphModule 和待融合列表

        # --- 断言 ---
        # 1. 检查融合后的模块参数
        fused_conv = _get_nested_module(gm_fused, "0")
        self.assertIsInstance(fused_conv, nn.Conv2d)
        torch.testing.assert_close(fused_conv.weight, expected_w)
        self.assertIsNotNone(fused_conv.bias)  # 确保有偏置
        torch.testing.assert_close(fused_conv.bias, expected_b)

        # 2. 检查 BN 模块是否已删除
        self.assertIsNone(_get_nested_module(gm_fused, "1"), "BatchNorm模块应已删除")
        self.assertNotIn(
            "1", dict(gm_fused.named_modules()), "BatchNorm FQN '1' 不应在模块字典中"
        )

        # 3. 检查图结构
        graph = gm_fused.graph
        bn_node_found = False
        conv_node = None
        for node in graph.nodes:
            if node.op == "call_module":
                if node.target == "1":
                    bn_node_found = True
                if node.target == "0":
                    conv_node = node
        self.assertFalse(bn_node_found, "图中不应再有调用 BN 模块 '1' 的节点")
        self.assertIsNotNone(conv_node, "图中应保留 Conv 模块 '0' 的节点")
        # 检查 Conv 节点的输出是否是图的最终输出（在这个简单模型中是）
        output_node = next(n for n in graph.nodes if n.op == "output")
        self.assertEqual(
            output_node.args[0], conv_node, "图的输出应直接来自融合后的 Conv 节点"
        )

        # 4. 检查数值输出 (融合前后应接近)
        output_orig = model_orig(self.default_input_conv)
        output_fused = gm_fused(self.default_input_conv)
        torch.testing.assert_close(output_orig, output_fused, rtol=1e-4, atol=1e-5)

    def test_fuse_linear_bn1d_noaffine_nobias(self):
        """测试融合 Linear (无偏置) 和 BatchNorm1d (affine=False)"""
        model_orig = nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.BatchNorm1d(20, affine=False),  # affine=False -> gamma=1, beta=0
        )
        # 模拟训练填充 BN 统计数据
        model_orig.train()
        for _ in range(2):
            model_orig(torch.randn(4, 10))
        model_orig.eval()

        # 计算预期结果
        linear_orig = model_orig[0]
        bn_orig = model_orig[1]
        expected_w, expected_b = calculate_fused_params(linear_orig, bn_orig)

        # 追踪和融合
        gm = self._trace_model(model_orig)
        gm_fused = fuse_conv_linear_bn_fx(gm, [["0", "1"]])

        # --- 断言 ---
        # 1. 检查融合后的模块参数
        fused_linear = _get_nested_module(gm_fused, "0")
        self.assertIsInstance(fused_linear, nn.Linear)
        torch.testing.assert_close(fused_linear.weight, expected_w)
        self.assertIsNotNone(fused_linear.bias)  # 融合后应该 *有* 偏置了
        torch.testing.assert_close(fused_linear.bias, expected_b)

        # 2. 检查 BN 模块是否已删除
        self.assertIsNone(_get_nested_module(gm_fused, "1"))
        self.assertNotIn("1", dict(gm_fused.named_modules()))

        # 3. 检查图结构 (类似 Conv 测试)
        graph = gm_fused.graph
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        self.assertFalse(bn_node_found)
        linear_node = next(
            node
            for node in graph.nodes
            if node.op == "call_module" and node.target == "0"
        )
        output_node = next(n for n in graph.nodes if n.op == "output")
        self.assertEqual(output_node.args[0], linear_node)

        # 4. 检查数值输出
        output_orig = model_orig(self.default_input_linear)
        output_fused = gm_fused(self.default_input_linear)
        torch.testing.assert_close(output_orig, output_fused, rtol=1e-4, atol=1e-5)

    def test_no_modules_to_fuse(self):
        """测试当传入空的融合列表时，函数行为正常"""
        model_orig = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU())
        gm = self._trace_model(model_orig)
        gm_copy = copy.deepcopy(gm)
        # 调用融合函数，传入空列表
        gm_fused = fuse_conv_linear_bn_fx(gm, [])

        # --- 断言 ---
        # 1. 模型对象应是同一个
        self.assertIs(gm_fused, gm)

        # 2. fuse_conv_linear_bn_fx 不应修改输入
        self.assertEqual(gm.code, gm_copy.code)

        # 3. 检查模块结构是否未改变
        original_module_names = {name for name, _ in gm_copy.named_modules()}
        fused_module_names = {name for name, _ in gm_fused.named_modules()}
        self.assertEqual(original_module_names, fused_module_names)

        original_module_types = {
            name: type(mod) for name, mod in gm_copy.named_modules() if name
        }
        fused_module_types = {
            name: type(mod) for name, mod in gm_fused.named_modules() if name
        }
        self.assertEqual(original_module_types, fused_module_types)

        self.assertTrue(isinstance(gm_copy, torch.fx.GraphModule))
        self.assertTrue(isinstance(gm_fused, torch.fx.GraphModule))

        # 4. 检查参数是否未改变
        for name, param in gm_fused.named_parameters():
            torch.testing.assert_close(param, gm_copy.get_parameter(name))

        # 5. 检查图结构是否未改变
        self.assertEqual(str(gm_fused.graph), str(gm_copy.graph))

    def test_fuse_multiple_pairs(self):
        """测试融合模型中的多个 Conv-BN 对"""
        model_orig = nn.Sequential(
            nn.Conv2d(3, 8, 1, bias=False),  # 0
            nn.BatchNorm2d(8, affine=True),  # 1
            nn.ReLU(),  # 2
            nn.Conv2d(8, 4, 1, bias=True),  # 3
            nn.BatchNorm2d(4, affine=False),  # 4
        )
        # 训练填充统计数据
        model_orig.train()
        for _ in range(2):
            model_orig(torch.randn(4, 3, 8, 8))
        model_orig.eval()

        # 计算预期结果
        expected_w0, expected_b0 = calculate_fused_params(model_orig[0], model_orig[1])
        expected_w3, expected_b3 = calculate_fused_params(model_orig[3], model_orig[4])

        # 追踪和融合
        gm = self._trace_model(model_orig)
        # 注意模块名称可能因为 deepcopy 变化，确保使用 gm 的名称
        modules_to_fuse_names = [["0", "1"], ["3", "4"]]
        gm_fused = fuse_conv_linear_bn_fx(gm, modules_to_fuse_names)

        # --- 断言 ---
        # 1. 检查第一个融合对 ('0')
        fused_conv0 = _get_nested_module(gm_fused, "0")
        self.assertIsInstance(fused_conv0, nn.Conv2d)
        torch.testing.assert_close(fused_conv0.weight, expected_w0)
        self.assertIsNotNone(fused_conv0.bias)
        torch.testing.assert_close(fused_conv0.bias, expected_b0)
        self.assertIsNone(_get_nested_module(gm_fused, "1"), "模块 '1' 应被删除")

        # 2. 检查第二个融合对 ('3')
        fused_conv3 = _get_nested_module(gm_fused, "3")
        self.assertIsInstance(fused_conv3, nn.Conv2d)
        torch.testing.assert_close(fused_conv3.weight, expected_w3)
        self.assertIsNotNone(fused_conv3.bias)
        torch.testing.assert_close(fused_conv3.bias, expected_b3)
        self.assertIsNone(_get_nested_module(gm_fused, "4"), "模块 '4' 应被删除")

        # 3. 检查 ReLU ('2') 是否还在
        relu_module = _get_nested_module(gm_fused, "2")
        self.assertIsInstance(relu_module, nn.ReLU)

        # 4. 检查图结构中 BN 节点是否都消失
        graph = gm_fused.graph
        bn1_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        bn4_node_found = any(
            node.op == "call_module" and node.target == "4" for node in graph.nodes
        )
        self.assertFalse(bn1_node_found, "图中不应有 BN 节点 '1'")
        self.assertFalse(bn4_node_found, "图中不应有 BN 节点 '4'")

        # 5. 检查数值输出
        input_tensor = torch.randn(1, 3, 8, 8)
        output_orig = model_orig(input_tensor)
        output_fused = gm_fused(input_tensor)
        torch.testing.assert_close(output_orig, output_fused, rtol=1e-4, atol=1e-5)


# --- 运行测试 ---
if __name__ == "__main__":
    unittest.main()
