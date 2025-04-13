# examples/quantize_toy_model_prepare.py
import torch
import torch.nn as nn
from micronet.compression.fx.quantization.core.quantizer import Quantizer
from micronet.compression.fx.quantization.core.qconfig import (
    default_placeholder_ptq_qconfig,
    default_placeholder_qat_qconfig,
)


# 1. 定义一个简单的模型
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 7 * 7, 10)  # 假设输入是 1x7x7

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.conv2(y)
        y = x + y  # 添加残差连接
        y = self.relu2(y)
        y = self.flatten(y)
        y = self.fc(y)
        return y


# 2. 创建模型实例
float_model = ToyModel().eval()  # 确保是 eval 模式进行追踪

# 3. 准备 PTQ (使用 Static QConfig)
# ptq_quantizer = Quantizer(qconfig=default_placeholder_ptq_qconfig)
ptq_quantizer = Quantizer(qconfig=default_placeholder_ptq_qconfig, debug=True)
prepared_ptq_model = ptq_quantizer.prepare(float_model)

# # 4. 准备 QAT (使用 QAT QConfig)
# float_model_for_qat = ToyModel().train()  # QAT 需要模型在 train 模式
# # qat_quantizer = Quantizer(qconfig=default_placeholder_qat_qconfig)
# qat_quantizer = Quantizer(qconfig=default_placeholder_qat_qconfig, debug=True)
# prepared_qat_model = qat_quantizer.prepare(float_model_for_qat)  # 传入 train 模式的模型

# 5. (可选) 测试准备后的模型是否能前向传播
print("\n测试 PTQ 准备后模型的前向传播...")
dummy_input = torch.randn(1, 1, 7, 7)
try:
    prepared_ptq_model.eval()  # PTQ 在 eval 模式
    output_ptq = prepared_ptq_model(dummy_input)
    print(f"PTQ 准备模型输出形状: {output_ptq.shape}")
except Exception as e:
    print(f"PTQ 准备模型前向传播失败: {e}")

# print("\n测试 QAT 准备后模型的前向传播...")
# try:
#     prepared_qat_model.train()  # QAT 在 train 模式
#     output_qat = prepared_qat_model(dummy_input)
#     print(f"QAT 准备模型输出形状: {output_qat.shape}")
# except Exception as e:
#     print(f"QAT 准备模型前向传播失败: {e}")
