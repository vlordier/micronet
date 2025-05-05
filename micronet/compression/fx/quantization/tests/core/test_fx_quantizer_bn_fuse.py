# micronet/compression/fx/quantization/tests/core/test_fx_quantizer_bn_fuse.py

import pytest
import copy

import torch
import torch.nn as nn

from micronet.compression.fx.quantization.core.quantizer import Quantizer
from micronet.compression.fx.quantization.core.qconfig import (
    QConfig,
    default_qconfig,
)
from micronet.compression.fx.quantization.core.fake_quant import FakeQuantize


# --- 测试类 ---
class TestQuantizerBnFuse:

    # 使用 pytest fixture 提供配置和默认输入
    @pytest.fixture
    def qconfig(self):
        return default_qconfig

    @pytest.fixture
    def default_input_conv(self):
        return torch.randn(1, 3, 16, 16)

    @pytest.fixture
    def default_input_linear(self):
        return torch.randn(1, 10)

    def _create_conv_bn_model(self):
        torch.manual_seed(1)  # 在创建模型时设置种子
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
        torch.manual_seed(1)  # 在创建模型时设置种子
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
        torch.manual_seed(1)  # 在创建模型时设置种子
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8)  # 0  # 1  # 2
        )
        model.eval()  # 不需要训练 BN，因为它不会被融合
        return model

    # --- 测试方法 (注入 fixtures) ---
    def test_fusion_enabled_conv_bn(self, qconfig, default_input_conv):  # 注入
        """测试 Quantizer.prepare 在 fuse_bn=True 时融合 Conv-BN"""
        model_orig = self._create_conv_bn_model()
        model_copy_for_compare = copy.deepcopy(model_orig)  # 用于比较数值输出
        quantizer = Quantizer(qconfig, fuse_bn=True, debug=False)  # 使用 fixture
        prepared_model = quantizer.prepare(model_orig)

        # 1. 检查模块结构：BN ('1') 应该消失，Conv ('0') 应该还在
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        assert "0" in final_modules, "融合后的 Conv ('0') 应该存在"
        assert isinstance(final_modules["0"], nn.Conv2d)
        assert "1" not in final_modules, "BatchNorm ('1') 应该已被融合移除"

        # 2. 检查是否有量化器插入
        found_observer = False
        for name, module in final_modules.items():
            if isinstance(module, FakeQuantize):
                found_observer = True
                break
        assert found_observer, "模型中应包含插入的 Observer/FakeQuantize 模块"
        # 更具体地检查图
        graph = prepared_model.graph
        observer_calls = [
            n.target
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        ]
        assert any(str(s).startswith("act_fake_quant_input") for s in observer_calls)
        assert any(str(s).startswith("weight_fake_quant_0") for s in observer_calls)
        assert any(
            str(s).startswith("act_fake_quant_after_mod_0") for s in observer_calls
        )

        # 3. 检查图中 BN 节点是否消失
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        assert not bn_node_found, "图中不应再有调用 BN 模块 '1' 的节点"

        # 4. 检查数值输出 (与仅融合版本比较)
        temp_quantizer = Quantizer(QConfig(None, None), fuse_bn=True)  # 无观察器配置
        fused_only_model = temp_quantizer.prepare(
            model_orig
        )  # 注意：这里 model_orig 已被第一次 prepare 修改，重新创建
        model_orig_for_fusion = self._create_conv_bn_model()
        fused_only_model = temp_quantizer.prepare(model_orig_for_fusion)
        output_orig = model_copy_for_compare(default_input_conv)  # 使用 fixture
        output_fused_only = fused_only_model(default_input_conv)  # 使用 fixture
        torch.testing.assert_close(output_orig, output_fused_only, rtol=1e-4, atol=1e-5)

    def test_fusion_enabled_linear_bn(self, qconfig, default_input_linear):  # 注入
        """测试 Quantizer.prepare 在 fuse_bn=True 时融合 Linear-BN"""
        model_orig = self._create_linear_bn_model()
        model_copy_for_compare = copy.deepcopy(model_orig)
        quantizer = Quantizer(qconfig, fuse_bn=True, debug=False)  # 使用 fixture
        prepared_model = quantizer.prepare(model_orig)

        # 1. 检查模块结构
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        assert "0" in final_modules
        assert isinstance(final_modules["0"], nn.Linear)
        assert "1" not in final_modules

        # 2. 检查量化器插入
        assert any(isinstance(m, FakeQuantize) for m in final_modules.values())
        graph = prepared_model.graph
        observer_calls = [
            n.target
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        ]
        assert any(str(s).startswith("act_fake_quant_input") for s in observer_calls)
        assert any(str(s).startswith("weight_fake_quant_0") for s in observer_calls)
        assert any(
            str(s).startswith("act_fake_quant_after_mod_0") for s in observer_calls
        )

        # 3. 检查图中 BN 节点消失
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        assert not bn_node_found

        # 4. 检查数值输出 (与仅融合版本比较)
        temp_quantizer = Quantizer(QConfig(None, None), fuse_bn=True)
        model_orig_for_fusion = self._create_linear_bn_model()  # 重新创建
        fused_only_model = temp_quantizer.prepare(model_orig_for_fusion)
        output_orig = model_copy_for_compare(default_input_linear)  # 使用 fixture
        output_fused_only = fused_only_model(default_input_linear)  # 使用 fixture
        torch.testing.assert_close(output_orig, output_fused_only, rtol=1e-4, atol=1e-5)

    def test_fusion_enabled_no_pattern(self, qconfig):  # 注入
        """测试 Quantizer.prepare 在 fuse_bn=True 但无 C-BN 模式时不融合"""
        model_orig = self._create_no_fuse_pattern_model()
        quantizer = Quantizer(qconfig, fuse_bn=True, debug=False)  # 使用 fixture
        prepared_model = quantizer.prepare(model_orig)

        # 1. 检查模块结构
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        assert "0" in final_modules
        assert isinstance(final_modules["0"], nn.Conv2d)
        assert "1" in final_modules
        assert isinstance(final_modules["1"], nn.ReLU)
        assert "2" in final_modules
        assert isinstance(final_modules["2"], nn.BatchNorm2d)

        # 2. 检查量化器插入
        assert any(isinstance(m, FakeQuantize) for m in final_modules.values())
        graph = prepared_model.graph
        observer_calls = {
            str(n.target)
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        }
        assert any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        assert any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        assert any(s.startswith("act_fake_quant_after_mod_1") for s in observer_calls)
        assert any(s.startswith("act_fake_quant_after_mod_2") for s in observer_calls)

        # 3. 检查图中 BN 节点 ('2') 仍然存在
        bn_node_found = any(
            node.op == "call_module" and node.target == "2" for node in graph.nodes
        )
        assert bn_node_found, "图中应保留 BN 模块 '2' 的节点"

    def test_fusion_disabled(self, qconfig):  # 注入
        """测试 Quantizer.prepare 在 fuse_bn=False 时即使有 C-BN 模式也不融合"""
        model_orig = self._create_conv_bn_model()  # 使用可融合的模型
        quantizer = Quantizer(
            qconfig, fuse_bn=False, debug=False
        )  # 禁用融合, 使用 fixture
        prepared_model = quantizer.prepare(model_orig)

        # 1. 检查模块结构
        final_modules = dict(prepared_model.named_modules(remove_duplicate=False))
        assert "0" in final_modules
        assert isinstance(final_modules["0"], nn.Conv2d)
        assert "1" in final_modules
        assert isinstance(final_modules["1"], nn.BatchNorm2d)

        # 2. 检查量化器插入
        assert any(isinstance(m, FakeQuantize) for m in final_modules.values())
        graph = prepared_model.graph
        observer_calls = {
            str(n.target)
            for n in graph.nodes
            if n.op == "call_module"
            and isinstance(final_modules.get(str(n.target)), FakeQuantize)
        }
        assert any(s.startswith("weight_fake_quant_0") for s in observer_calls)
        assert any(s.startswith("act_fake_quant_after_mod_0") for s in observer_calls)
        assert any(s.startswith("act_fake_quant_after_mod_1") for s in observer_calls)

        # 3. 检查图中 BN 节点 ('1') 仍然存在
        bn_node_found = any(
            node.op == "call_module" and node.target == "1" for node in graph.nodes
        )
        assert bn_node_found, "图中应保留 BN 模块 '1' 的节点"
