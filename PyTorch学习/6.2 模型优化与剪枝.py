# PyTorch模型优化与剪枝学习
# 主要内容：模型剪枝、量化、知识蒸馏

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.nn.utils import prune

print("=== 创建示例模型 ===")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
print(f"原始模型:\n{model}")

print("\n=== L1范数剪枝 ===")
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)
prune.l1_unstructured(model.fc2, name='weight', amount=0.3)
print(f"剪枝后fc1权重稀疏度: {(model.fc1.weight == 0).float().mean().item():.4f}")

print("\n=== 结构化剪枝 ===")
prune.ln_structured(model.conv1, name='weight', amount=0.2, n=1, dim=0)
print(f"剪枝后conv1输出通道数: {model.conv1.weight.shape[0]}")

print("\n=== 移除剪枝参数 ===")
prune.remove(model.fc1, 'weight')
prune.remove(model.fc2, 'weight')
print("剪枝参数已移除")

print("\n=== 模型量化 ===")
class QuantizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dequant = DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

q_model = QuantizedModel()
q_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
q_model = torch.quantization.prepare(q_model)

print("\n=== 模拟量化 ===")
input_data = torch.randn(1, 1, 28, 28)
output = q_model(input_data)
print(f"量化模型输出形状: {output.shape}")

print("\n=== 知识蒸馏 ===")
teacher_model = SimpleModel()
student_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

def distillation_loss(student_output, teacher_output, labels, alpha=0.5, temperature=3.0):
    hard_loss = F.cross_entropy(student_output, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_output / temperature, dim=1),
        F.softmax(teacher_output / temperature, dim=1),
        reduction='batchmean'
    ) * temperature**2
    return alpha * hard_loss + (1 - alpha) * soft_loss

print("知识蒸馏损失函数已定义")