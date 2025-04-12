# quantization_framework/configs/default_static_qconfig.py (更新后)

from micronet.compression.fx.quantization.core.qconfig import QConfig
from micronet.compression.fx.quantization.core.observer import PlaceholderObserver

# 使用占位符 Observer
default_static_qconfig = QConfig(
    activation=PlaceholderObserver,  # PTQ 使用 Observer
    weight=PlaceholderObserver,  # PTQ 使用 Observer
)

print("Loaded default_static_qconfig with PlaceholderObserver")
