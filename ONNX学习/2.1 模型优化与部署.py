# ONNX模型优化与部署学习
# 主要内容：模型量化、模型剪枝、跨框架转换、部署示例

import torch
import torch.nn as nn
import numpy as np

print("=== 创建PyTorch模型 ===")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleModel()
model.eval()
print("PyTorch模型已创建")

print("\n=== 转换为ONNX ===")
dummy_input = torch.randn(1, 3, 32, 32)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print("模型已转换为ONNX格式")

print("\n=== 验证ONNX模型 ===")
import onnx
import onnxruntime as ort

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过")

print("\n=== ONNX Runtime推理 ===")
ort_session = ort.InferenceSession("model.onnx")
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

input_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
ort_output = ort_session.run([output_name], {input_name: input_data})[0]
print(f"ONNX输出形状: {ort_output.shape}")

print("\n=== 模型量化 ===")
from onnxruntime.quantization import quantize_dynamic, QuantType

quantized_model = quantize_dynamic(
    "model.onnx",
    "model_quantized.onnx",
    weight_type=QuantType.QUInt8
)
print("模型量化完成")

print("\n=== 比较模型大小 ===")
import os
original_size = os.path.getsize("model.onnx") / (1024 * 1024)
quantized_size = os.path.getsize("model_quantized.onnx") / (1024 * 1024)
print(f"原始模型大小: {original_size:.2f} MB")
print(f"量化模型大小: {quantized_size:.2f} MB")
print(f"压缩比: {original_size / quantized_size:.2f}x")

print("\n=== PyTorch转ONNX再转TensorFlow ===")
try:
    import onnx_tf
    
    onnx_model = onnx.load("model.onnx")
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph("model_tf")
    print("已转换为TensorFlow格式")
except ImportError:
    print("onnx-tf未安装，跳过TensorFlow转换")

print("\n=== ONNX模型可视化 ===")
try:
    import netron
    netron.start("model.onnx")
    print("模型可视化已启动")
except:
    print("Netron未安装，跳过可视化")

print("\n=== 清理测试文件 ===")
for f in ["model.onnx", "model_quantized.onnx"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")