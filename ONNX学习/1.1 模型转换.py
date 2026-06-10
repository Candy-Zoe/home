# ONNX模型转换学习
# 主要内容：PyTorch转ONNX、TensorFlow转ONNX、模型验证

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

print("=== 创建PyTorch模型 ===")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
model.eval()

print("\n=== 导出为ONNX ===")
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
print("ONNX模型已导出")

print("\n=== 加载ONNX模型 ===")
onnx_model = onnx.load('model.onnx')
print(f"输入节点: {[node.name for node in onnx_model.graph.input]}")
print(f"输出节点: {[node.name for node in onnx_model.graph.output]}")

print("\n=== 验证ONNX模型 ===")
try:
    onnx.checker.check_model(onnx_model)
    print("ONNX模型验证通过")
except Exception as e:
    print(f"ONNX模型验证失败: {e}")

print("\n=== 使用ONNX Runtime推理 ===")
ort_session = ort.InferenceSession('model.onnx')
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

input_data = dummy_input.numpy()
result = ort_session.run([output_name], {input_name: input_data})
print(f"ONNX推理结果形状: {result[0].shape}")
print(f"ONNX推理结果: {result[0]}")

print("\n=== 对比PyTorch和ONNX结果 ===")
with torch.no_grad():
    torch_result = model(dummy_input).numpy()

print(f"PyTorch结果: {torch_result}")
print(f"ONNX结果: {result[0]}")
print(f"结果差异: {np.abs(torch_result - result[0]).max()}")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('model.onnx'):
    os.remove('model.onnx')
    print("已删除测试文件")

import numpy as np