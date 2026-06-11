# ONNX模型转换学习
# 主要内容：PyTorch/TensorFlow模型转换为ONNX、ONNX模型推理

# 导入必要的库
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# 创建PyTorch模型
print("=== 创建PyTorch模型 ===")

class SimpleCNN(nn.Module):
    """简单的CNN模型"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleCNN()
model.eval()  # 设置为评估模式

print(f"模型结构:\n{model}")

# 创建示例输入
input_shape = (1, 1, 28, 28)  # (batch_size, channels, height, width)
dummy_input = torch.randn(input_shape)

print(f"\n输入形状: {input_shape}")

# PyTorch模型转换为ONNX
print("\n=== PyTorch转ONNX ===")

# 导出ONNX模型
onnx_path = 'simple_cnn.onnx'

torch.onnx.export(
    model,                    # 要导出的模型
    dummy_input,              # 示例输入
    onnx_path,                # 输出路径
    export_params=True,       # 是否导出参数
    opset_version=13,        # ONNX算子集版本
    do_constant_folding=True, # 是否执行常量折叠优化
    input_names=['input'],    # 输入名称
    output_names=['output'],  # 输出名称
    dynamic_axes={           # 动态轴（支持可变批次大小）
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"ONNX模型已导出到: {onnx_path}")

# 验证ONNX模型
print("\n=== 验证ONNX模型 ===")

# 加载ONNX模型
onnx_model = onnx.load(onnx_path)

# 检查模型是否有效
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过！")

# 打印模型信息
print(f"\nONNX模型信息:")
print(f"  输入: {[input.name for input in onnx_model.graph.input]}")
print(f"  输出: {[output.name for output in onnx_model.graph.output]}")
print(f"  算子数量: {len(onnx_model.graph.node)}")

# ONNX模型推理
print("\n=== ONNX模型推理 ===")

# 创建ONNX Runtime会话
session = ort.InferenceSession(onnx_path)

# 获取输入输出名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"输入名称: {input_name}")
print(f"输出名称: {output_name}")

# 准备输入数据
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# 执行推理
output = session.run([output_name], {input_name: input_data})

print(f"输入形状: {input_data.shape}")
print(f"输出形状: {output[0].shape}")
print(f"输出示例（前5个概率）: {output[0][0][:5]}")

# 对比PyTorch和ONNX推理结果
print("\n=== 对比PyTorch和ONNX结果 ===")

# PyTorch推理
with torch.no_grad():
    torch_output = model(torch.from_numpy(input_data))

# ONNX推理
onnx_output = session.run([output_name], {input_name: input_data})[0]

# 计算差异
diff = np.abs(torch_output.numpy() - onnx_output)
print(f"最大差异: {diff.max():.6f}")
print(f"平均差异: {diff.mean():.6f}")

if diff.max() < 1e-5:
    print("PyTorch和ONNX推理结果一致！")
else:
    print("PyTorch和ONNX推理结果存在差异")

# 动态批次大小测试
print("\n=== 动态批次大小测试 ===")

# 使用不同批次大小
batch_sizes = [1, 2, 4, 8]

for batch_size in batch_sizes:
    # 创建不同批次大小的输入
    input_data = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    
    # ONNX推理
    output = session.run([output_name], {input_name: input_data})
    
    print(f"批次大小 {batch_size}: 输出形状 {output[0].shape}")

# TensorFlow模型转ONNX
print("\n=== TensorFlow模型转ONNX ===")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    
    # 创建TensorFlow模型
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    
    tf_model = Model(inputs, outputs)
    tf_model.compile(optimizer='adam', loss='categorical_crossentropy')
    
    print("TensorFlow模型创建完成")
    
    # 尝试转换（需要安装tf2onnx）
    try:
        import tf2onnx
        
        tf_onnx_path = 'tf_simple_cnn.onnx'
        tf_model.save('tf_model.h5')
        
        # 转换为ONNX
        spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
        tf2onnx.convert.from_keras(tf_model, input_signature=spec, output_path=tf_onnx_path)
        
        print(f"TensorFlow模型已转换为ONNX: {tf_onnx_path}")
        
        # 验证转换后的模型
        tf_onnx_model = onnx.load(tf_onnx_path)
        onnx.checker.check_model(tf_onnx_model)
        print("TensorFlow转ONNX模型验证通过！")
        
        # 清理文件
        import os
        if os.path.exists('tf_model.h5'):
            os.remove('tf_model.h5')
        if os.path.exists(tf_onnx_path):
            os.remove(tf_onnx_path)
            
    except ImportError:
        print("tf2onnx未安装，跳过TensorFlow转ONNX")
        
except ImportError:
    print("TensorFlow未安装，跳过TensorFlow示例")

# ONNX模型优化
print("\n=== ONNX模型优化 ===")

try:
    from onnxruntime.quantization import quantize_dynamic
    
    # 量化模型
    quantized_path = 'simple_cnn_quantized.onnx'
    quantize_dynamic(onnx_path, quantized_path)
    
    print(f"量化模型已保存到: {quantized_path}")
    
    # 比较模型大小
    import os
    original_size = os.path.getsize(onnx_path) / 1024
    quantized_size = os.path.getsize(quantized_path) / 1024
    
    print(f"原始模型大小: {original_size:.2f} KB")
    print(f"量化后模型大小: {quantized_size:.2f} KB")
    print(f"压缩比例: {(1 - quantized_size / original_size) * 100:.1f}%")
    
    # 测试量化模型性能
    quantized_session = ort.InferenceSession(quantized_path)
    quantized_output = quantized_session.run([output_name], {input_name: input_data})
    
    # 计算准确率损失
    quantized_diff = np.abs(output[0] - quantized_output[0])
    print(f"量化后最大差异: {quantized_diff.max():.6f}")
    
    # 清理量化模型
    if os.path.exists(quantized_path):
        os.remove(quantized_path)
        
except ImportError:
    print("ONNX Runtime量化工具未安装，跳过量化")

# 清理ONNX文件
if os.path.exists(onnx_path):
    os.remove(onnx_path)
    print(f"\n已删除: {onnx_path}")

print("\nONNX模型转换学习完成！")