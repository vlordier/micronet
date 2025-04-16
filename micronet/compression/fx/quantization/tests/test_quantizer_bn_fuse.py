import unittest
import copy

import torch
import torch.nn as nn

from micronet.compression.fx.quantization.core.quantizer import Quantizer
from micronet.compression.fx.quantization.core.qconfig import (
    QConfig,
    default_qconfig,
)
from micronet.compression.fx.quantization.core.fake_quant import FakeQuantize


class TestQuantizerBnFuse(unittest.TestCase):

    def setUp(self):
        self.qconfig = default_qconfig
        self.default_input_conv = torch.randn(1, 3, 16, 16)
        self.default_input_linear = torch.randn(1, 10)
        torch.manual_seed(1)

    def _create_conv_bn_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True),  # 0
            nn.BatchNorm2d(8, affine=True),  # 1
        )
        # 模拟训练
        model.train()
        for _ in range(2):
            model(torch.randn(4, 3, 16, 16))
        model.eval()
        return model

    def _create_linear_bn_model(self):
        model = nn.Sequential(
            nn.Linear(10, 20, bias=True), nn.BatchNorm1d(20, affine=True)  # 0  # 1
        )
        # 模拟训练
        model.train()
        for _ in range(2):
            model(torch.randn(4, 10))
        model.eval()
        return model

    def _create_no_fuse_pattern_model(self):
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8)  # 0  # 1  # 2
        )
        model.eval()  # 不需要训练 BN，因为它不会被融合
        return model

    def test_fusion_enabled_conv_bn(self):
        """测试 Quantizer.prepare 在 fuse_bn=True 时融合 Conv-BN"""
        model_orig = self._create_conv_bn_model()
        model_copy_for_compare = copy.deepcopy(model_orig)  # 用于比较数值输出
        quantizer = Quantizer(self.qconfig, fuse_bn=True, debug=False)
        prepared_model = quantizer.prepare(model_orig)

        # --- 断言 ---
        # 1. 检查模块结构：BN ('1') 应该消失，Conv ('0') 应该还在
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        # print("Final modules:", list(final_modules.keys())) # 调试用
        self.assertIn("0", final_modules, "融合后的 Conv ('0') 应该存在")
        self.assertIsInstance(final_modules["0"], nn.Conv2d)
        self.assertNotIn("1", final_modules, "BatchNorm ('1') 应该已被融合移除")

        # 2. 检查是否有量化器插入（至少应该有输入和 Conv 输出的观察器）
        found_observer = False
        for name, module in final_modules.items():
            if isinstance(module, FakeQuantize):
                found_observer = True
                break
        self.assertTrue(found_observer, "模型中应包含插入的 Observer 模块")
        # 更具体地检查图
        graph = prepared_model.graph
        # 应该有对 act_fake_quant_input_... 的调用 (对输入)
        # 应该有对 weight_fake_quant_... 的调用 (对 conv 权重)
        # 应该有对 act_fake_quant_after_mod_0... 的调用 (对 conv 输出)
        observer_calls = [
            n.target
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        ]
        self.assertTrue(
            any(s.startswith("act_fake_quant_input") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        )

        # 3. 检查图中 BN 节点是否消失
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        self.assertFalse(bn_node_found, "图中不应再有调用 BN 模块 '1' 的节点")

        # 4. 检查数值输出 (Quantizer prepare 后带 observer 的模型 vs 原始 eval 模型)
        # 注意：带 observer 的模型输出可能与原始不同，但融合本身应该保持数值一致性
        # 我们比较融合后的模型（无 observer）与原始模型
        # 创建一个仅融合无观察器的版本来比较数值
        temp_quantizer = Quantizer(QConfig(None, None), fuse_bn=True)  # 无观察器配置
        fused_only_model = temp_quantizer.prepare(model_orig)
        output_orig = model_copy_for_compare(self.default_input_conv)
        output_fused_only = fused_only_model(self.default_input_conv)
        torch.testing.assert_close(output_orig, output_fused_only, rtol=1e-4, atol=1e-5)

    def test_fusion_enabled_linear_bn(self):
        """测试 Quantizer.prepare 在 fuse_bn=True 时融合 Linear-BN"""
        model_orig = self._create_linear_bn_model()
        model_copy_for_compare = copy.deepcopy(model_orig)
        quantizer = Quantizer(self.qconfig, fuse_bn=True, debug=False)
        prepared_model = quantizer.prepare(model_orig)

        # --- 断言 ---
        # 1. 检查模块结构：BN ('1') 消失，Linear ('0') 还在
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        self.assertIn("0", final_modules)
        self.assertIsInstance(final_modules["0"], nn.Linear)
        self.assertNotIn("1", final_modules)

        # 2. 检查量化器插入
        self.assertTrue(
            any(isinstance(m, FakeQuantize) for m in final_modules.values())
        )
        graph = prepared_model.graph
        observer_calls = [
            n.target
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        ]
        self.assertTrue(
            any(s.startswith("act_fake_quant_input") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        )

        # 3. 检查图中 BN 节点消失
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        self.assertFalse(bn_node_found)

        # 4. 检查数值输出 (与仅融合版本比较)
        temp_quantizer = Quantizer(QConfig(None, None), fuse_bn=True)
        fused_only_model = temp_quantizer.prepare(model_orig)
        output_orig = model_copy_for_compare(self.default_input_linear)
        output_fused_only = fused_only_model(self.default_input_linear)
        torch.testing.assert_close(output_orig, output_fused_only, rtol=1e-4, atol=1e-5)

    def test_fusion_enabled_no_pattern(self):
        """测试 Quantizer.prepare 在 fuse_bn=True 但无 C-BN 模式时不融合"""
        model_orig = self._create_no_fuse_pattern_model()
        quantizer = Quantizer(self.qconfig, fuse_bn=True, debug=False)
        prepared_model = quantizer.prepare(model_orig)

        # --- 断言 ---
        # 1. 检查模块结构：Conv('0'), ReLU('1'), BN('2') 都应该还在 (除了被 observer 替换)
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        # print("Final modules (no pattern):", list(final_modules.keys()))
        self.assertIn("0", final_modules)
        self.assertIsInstance(final_modules["0"], nn.Conv2d)
        self.assertIn("1", final_modules)
        self.assertIsInstance(final_modules["1"], nn.ReLU)
        self.assertIn("2", final_modules)
        self.assertIsInstance(final_modules["2"], nn.BatchNorm2d)

        # 2. 检查量化器插入 (应该在 Conv 输出, ReLU 输出, BN 输出等地方有)
        self.assertTrue(
            any(isinstance(m, FakeQuantize) for m in final_modules.values())
        )
        graph = prepared_model.graph
        observer_calls = {
            str(n.target)
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        }
        # print("Observer calls (no pattern):", observer_calls)
        # 应该观察 conv(0) 的权重和输出，relu(1) 的输出，bn(2) 的输出
        self.assertTrue(
            any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_1") for s in observer_calls)
        )  # After ReLU
        # BN 默认也会被观察
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_2") for s in observer_calls)
        )  # After BN

        # 3. 检查图中 BN 节点 ('2') 仍然存在
        bn_node_found = any(
            node.op == "call_module" and node.target == "2" for node in graph.nodes
        )
        self.assertTrue(bn_node_found, "图中应保留 BN 模块 '2' 的节点")

    def test_fusion_disabled(self):
        """测试 Quantizer.prepare 在 fuse_bn=False 时即使有 C-BN 模式也不融合"""
        model_orig = self._create_conv_bn_model()  # 使用可融合的模型
        quantizer = Quantizer(self.qconfig, fuse_bn=False, debug=False)  # 禁用融合
        prepared_model = quantizer.prepare(model_orig)

        # --- 断言 ---
        # 1. 检查模块结构：Conv('0') 和 BN('1') 都应该还在
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        self.assertIn("0", final_modules)
        self.assertIsInstance(final_modules["0"], nn.Conv2d)
        self.assertIn("1", final_modules)
        self.assertIsInstance(final_modules["1"], nn.BatchNorm2d)

        # 2. 检查量化器插入 (应该在 Conv 输出, BN 输出等地方有)
        self.assertTrue(
            any(isinstance(m, FakeQuantize) for m in final_modules.values())
        )
        graph = prepared_model.graph
        observer_calls = {
            str(n.target)
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        }
        self.assertTrue(
            any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        )
        self.assertTrue(
            any(s.startswith("act_fake_quant_after_mod_1") for s in observer_calls)
        )  # After BN

        # 3. 检查图中 BN 节点 ('1') 仍然存在
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        self.assertTrue(bn_node_found, "图中应保留 BN 模块 '1' 的节点")


# --- 运行测试 ---
if __name__ == "__main__":
    unittest.main()
