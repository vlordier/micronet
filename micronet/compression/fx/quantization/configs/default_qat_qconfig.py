# quantization_framework/configs/default_qat_qconfig.py (更新后)

from micronet.compression.fx.quantization.core.qconfig import QConfig
from micronet.compression.fx.quantization.core.fake_quant import PlaceholderFakeQuant

# 使用占位符 FakeQuant
default_qat_qconfig = QConfig(
    activation=PlaceholderFakeQuant,  # QAT 使用 FakeQuant
    weight=PlaceholderFakeQuant,  # QAT 使用 FakeQuant
)

print("Loaded default_qat_qconfig with PlaceholderFakeQuant")
