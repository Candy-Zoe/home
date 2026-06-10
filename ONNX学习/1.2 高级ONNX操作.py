# ONNX高级操作学习
# 主要内容：模型优化、量化、自定义算子

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import onnxoptimizer

print("=== 创建模型 ===")
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 28 * 28, 10)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNNModel()
model.eval()

print("\n=== 导出ONNX模型 ===")
dummy_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    model,
    dummy_input,
    'cnn_model.onnx',
    export_params=True,
    opset_version=11
)
print("CNN模型已导出")

print("\n=== ONNX模型优化 ===")
onnx_model = onnx.load('cnn_model.onnx')
optimized_model = onnxoptimizer.optimize(onnx_model)
onnx.save(optimized_model, 'cnn_model_optimized.onnx')
print("模型已优化")

print("\n=== ONNX Runtime推理 ===")
ort_session = ort.InferenceSession('cnn_model_optimized.onnx')
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

input_data = dummy_input.numpy()
result = ort_session.run([output_name], {input_name: input_data})
print(f"推理结果形状: {result[0].shape}")

print("\n=== 量化模型 ===")
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    'cnn_model_optimized.onnx',
    'cnn_model_quantized.onnx',
    weight_type=QuantType.QUInt8
)
print("模型已量化")

print("\n=== 测试量化模型 ===")
ort_session_quantized = ort.InferenceSession('cnn_model_quantized.onnx')
result_quantized = ort_session_quantized.run([output_name], {input_name: input_data})
print(f"量化模型推理结果形状: {result_quantized[0].shape}")

print("\n=== 清理测试文件 ===")
import os
for f in ['cnn_model.onnx', 'cnn_model_optimized.onnx', 'cnn_model_quantized.onnx']:
    if os.path.exists(f):
        os.remove(f)
print("已删除测试文件")