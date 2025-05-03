# micronet/compression/fx/quantization/tests/core/test_qconfig.py

import pytest
import torch
import torch.nn as nn
import functools
from typing import Optional

from micronet.compression.fx.quantization.core.qconfig import (
    QConfig,
    QConfigMapping,
    FakeQuantize,
    MinMaxObserver,
    default_ptq_qconfig,
    default_qat_qconfig,
    default_qconfig,
    get_default_ptq_qconfig,
    get_default_qat_qconfig,
    QuantizerClsFactory,
)

# --- 辅助模块和配置 ---


# 定义一些简单的模块类型用于测试
class MyConv(nn.Module):
    pass


class MyLinear(nn.Module):
    pass


class MyActiv(nn.Module):
    pass


class SubModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = MyConv()
        self.linear = MyLinear()


class TopModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = SubModule()
        self.act = MyActiv()
        self.another_conv = MyConv()  # 用于测试同类型不同实例


# 创建一些不同的 QConfig 实例用于测试
# 注意：这里使用 lambda 只是为了快速创建不同的 callable 对象
# 实际中 default_qconfig 使用了 functools.partial，测试时会检查那个
qconfig_act_only = QConfig(activation=lambda: FakeQuantize(), weight=None)
qconfig_weight_only = QConfig(activation=None, weight=lambda: FakeQuantize())
qconfig_both = QConfig(activation=lambda: FakeQuantize(), weight=lambda: FakeQuantize())
qconfig_custom1 = QConfig(
    activation=lambda: "FakeQuantCustomAct1", weight=lambda: "FakeQuantCustomWgt1"
)
qconfig_custom2 = QConfig(
    activation=lambda: "FakeQuantCustomAct2", weight=lambda: "FakeQuantCustomWgt2"
)
qconfig_global_test = QConfig(
    activation=lambda: "GlobalAct", weight=lambda: "GlobalWgt"
)


# --- 测试 QConfig 和默认配置 ---


def test_qconfig_creation():
    """测试 QConfig 数据类的基本创建和属性。"""
    act_factory = lambda: FakeQuantize(dtype=torch.quint8)
    wgt_factory = lambda: FakeQuantize(dtype=torch.qint8)
    qc = QConfig(activation=act_factory, weight=wgt_factory)
    assert qc.activation is act_factory
    assert qc.weight is wgt_factory
    # 调用工厂函数检查是否能创建实例
    assert isinstance(qc.activation(), FakeQuantize)
    assert isinstance(qc.weight(), FakeQuantize)
    assert qc.activation().dtype == torch.quint8
    assert qc.weight().dtype == torch.qint8


def test_qconfig_creation_none():
    """测试 QConfig 允许 None 值。"""
    qc1 = QConfig(activation=None, weight=lambda: FakeQuantize())
    assert qc1.activation is None
    assert qc1.weight is not None

    qc2 = QConfig(activation=lambda: FakeQuantize(), weight=None)
    assert qc2.activation is not None
    assert qc2.weight is None

    qc3 = QConfig(activation=None, weight=None)
    assert qc3.activation is None
    assert qc3.weight is None


def test_default_qconfigs_exist():
    """检查默认 QConfig 实例是否存在且类型正确。"""
    assert isinstance(default_ptq_qconfig, QConfig)
    assert isinstance(default_qat_qconfig, QConfig)
    assert isinstance(default_qconfig, QConfig)
    assert default_qconfig is default_ptq_qconfig  # 检查默认指向 PTQ


def check_fakequant_factory(
    factory: Optional[QuantizerClsFactory],
    expected_dtype: torch.dtype,
    expected_qscheme: torch.qscheme,
    is_weight: bool,
):
    """辅助函数，检查 FakeQuantize 工厂函数的参数。"""
    assert factory is not None
    assert isinstance(factory, functools.partial)
    assert factory.func is FakeQuantize
    assert factory.keywords.get("observer_cls") is MinMaxObserver
    assert factory.keywords.get("dtype") == expected_dtype
    assert factory.keywords.get("qscheme") == expected_qscheme
    if is_weight:
        assert factory.keywords.get("reduce_range") == False  # 检查权重的 reduce_range
        # 权重通常对称，检查一下
        assert expected_qscheme == torch.per_tensor_symmetric
    else:
        # 激活通常仿射，检查一下
        assert expected_qscheme == torch.per_tensor_affine


def test_default_ptq_qconfig_details():
    """详细检查默认 PTQ QConfig 的工厂函数和参数。"""
    check_fakequant_factory(
        default_ptq_qconfig.activation,
        expected_dtype=torch.quint8,
        expected_qscheme=torch.per_tensor_affine,
        is_weight=False,
    )
    check_fakequant_factory(
        default_ptq_qconfig.weight,
        expected_dtype=torch.qint8,
        expected_qscheme=torch.per_tensor_symmetric,
        is_weight=True,
    )


def test_default_qat_qconfig_details():
    """详细检查默认 QAT QConfig 的工厂函数和参数（结构应与 PTQ 类似）。"""
    # QAT 配置在结构上通常与 PTQ 相同，区别在于 FakeQuantize 内部状态
    check_fakequant_factory(
        default_qat_qconfig.activation,
        expected_dtype=torch.quint8,
        expected_qscheme=torch.per_tensor_affine,
        is_weight=False,
    )
    check_fakequant_factory(
        default_qat_qconfig.weight,
        expected_dtype=torch.qint8,
        expected_qscheme=torch.per_tensor_symmetric,
        is_weight=True,
    )


def test_get_default_qconfigs():
    """测试 get_default_ 函数是否返回正确的对象。"""
    assert get_default_ptq_qconfig() is default_ptq_qconfig
    assert get_default_qat_qconfig() is default_qat_qconfig


# --- 测试 QConfigMapping ---


@pytest.fixture
def mapping() -> QConfigMapping:
    """提供一个空的 QConfigMapping 实例。"""
    return QConfigMapping()


def test_qconfig_mapping_initialization(mapping: QConfigMapping):
    """测试 QConfigMapping 初始化状态。"""
    assert mapping._global_qconfig is None
    assert mapping._module_type_qconfigs == {}
    assert mapping._object_name_qconfigs == {}


def test_qconfig_mapping_set_global(mapping: QConfigMapping):
    """测试设置和移除全局 QConfig。"""
    assert mapping.set_global(qconfig_global_test) is mapping  # 测试链式调用
    assert mapping._global_qconfig is qconfig_global_test
    mapping.set_global(None)
    assert mapping._global_qconfig is None


def test_qconfig_mapping_set_module_type(mapping: QConfigMapping):
    """测试按模块类型设置和移除 QConfig。"""
    assert mapping.set_module_type(MyConv, qconfig_custom1) is mapping  # 测试链式调用
    assert mapping._module_type_qconfigs == {MyConv: qconfig_custom1}
    mapping.set_module_type(MyLinear, qconfig_custom2)
    assert mapping._module_type_qconfigs == {
        MyConv: qconfig_custom1,
        MyLinear: qconfig_custom2,
    }
    mapping.set_module_type(MyConv, None)  # 移除 MyConv 的特定配置
    assert mapping._module_type_qconfigs == {MyConv: None, MyLinear: qconfig_custom2}
    mapping.set_module_type(MyLinear, None)  # 移除 MyLinear 的特定配置
    assert mapping._module_type_qconfigs == {MyConv: None, MyLinear: None}


def test_qconfig_mapping_set_object_name(mapping: QConfigMapping):
    """测试按对象名称设置和移除 QConfig。"""
    fqn1 = "sub.conv"
    fqn2 = "act"
    assert mapping.set_object_name(fqn1, qconfig_custom1) is mapping  # 测试链式调用
    assert mapping._object_name_qconfigs == {fqn1: qconfig_custom1}
    mapping.set_object_name(fqn2, qconfig_custom2)
    assert mapping._object_name_qconfigs == {
        fqn1: qconfig_custom1,
        fqn2: qconfig_custom2,
    }
    mapping.set_object_name(fqn1, None)  # 禁用 fqn1 的量化
    assert mapping._object_name_qconfigs == {fqn1: None, fqn2: qconfig_custom2}
    mapping.set_object_name(fqn2, None)  # 禁用 fqn2 的量化
    assert mapping._object_name_qconfigs == {fqn1: None, fqn2: None}


def test_qconfig_mapping_getitem(mapping: QConfigMapping):
    """测试使用方括号直接访问配置（不考虑优先级）和检查直接映射。"""
    mapping.set_module_type(MyConv, qconfig_custom1)
    mapping.set_object_name("sub.linear", qconfig_custom2)

    # 测试 __getitem__
    assert mapping[MyConv] is qconfig_custom1
    assert mapping["sub.linear"] is qconfig_custom2

    # 测试直接检查内部字典，确认未设置的键不存在
    assert mapping._module_type_qconfigs.get(MyLinear) is None
    assert mapping._object_name_qconfigs.get("sub.conv") is None

    # 测试无效类型
    with pytest.raises(TypeError, match="不支持的键类型: .*"):
        _ = mapping[123]

    # 测试 __getitem__ 对未设置键的行为
    with pytest.raises(KeyError):
        _ = mapping[MyActiv]  # 类型未设置
    with pytest.raises(KeyError):
        _ = mapping["nonexistent"]  # 名称未设置


def test_qconfig_mapping_get_qconfig_priority(mapping: QConfigMapping):
    """测试 get_qconfig 的优先级规则。"""
    fqn_conv = "sub.conv"
    fqn_linear = "sub.linear"
    fqn_activ = "act"

    # 设置各级配置
    mapping.set_global(qconfig_global_test)
    mapping.set_module_type(MyConv, qconfig_custom1)  # Conv 类型用 custom1
    mapping.set_module_type(MyLinear, None)  # Linear 类型禁用
    mapping.set_object_name(fqn_conv, qconfig_custom2)  # 特定 conv 实例用 custom2
    mapping.set_object_name(fqn_activ, None)  # 特定 act 实例禁用

    # 1. 对象名称优先
    assert (
        mapping.get_qconfig(MyConv, fqn_conv) is qconfig_custom2
    )  # 即使类型有配置，名称优先
    assert mapping.get_qconfig(MyActiv, fqn_activ) is None  # 名称指定禁用，优先

    # 2. 模块类型次之 (无对象名称或名称未命中)
    #    假设 'another_conv' 没有特定名称设置
    assert (
        mapping.get_qconfig(MyConv, "another_conv") is qconfig_custom1
    )  # 匹配类型 MyConv
    assert (
        mapping.get_qconfig(MyLinear, fqn_linear) is None
    )  # 匹配类型 MyLinear (被禁用)

    # 3. 全局配置最后 (无对象名称且无类型匹配)
    #    MyActiv 类型没有特定配置，fqn_activ 被禁用了，但假设我们查一个不同的 MyActiv 实例
    assert (
        mapping.get_qconfig(MyActiv, "another_act") is qconfig_global_test
    )  # 回退到全局

    # 4. 如果全局也没有，则为 None
    mapping.set_global(None)
    assert mapping.get_qconfig(MyActiv, "yet_another_act") is None  # 所有级别都无配置


def test_qconfig_mapping_get_qconfig_no_object_name(mapping: QConfigMapping):
    """测试 get_qconfig 在不提供 object_name 时的情况。"""
    mapping.set_global(qconfig_global_test)
    mapping.set_module_type(MyConv, qconfig_custom1)
    mapping.set_module_type(MyLinear, None)

    assert mapping.get_qconfig(MyConv) is qconfig_custom1  # 匹配类型
    assert mapping.get_qconfig(MyLinear) is None  # 匹配类型 (禁用)
    assert mapping.get_qconfig(MyActiv) is qconfig_global_test  # 回退到全局

    mapping.set_global(None)
    assert mapping.get_qconfig(MyActiv) is None  # 全局移除后为 None
