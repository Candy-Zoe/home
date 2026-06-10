# PyTorch迁移学习学习
# 主要内容：使用预训练模型进行迁移学习

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

print("=== 加载预训练模型 ===")
model = models.resnet18(pretrained=True)

print("\n=== 冻结特征提取层 ===")
for param in model.parameters():
    param.requires_grad = False

print("\n=== 修改分类层 ===")
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

print("\n=== 数据预处理 ===")
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("\n=== 定义损失和优化器 ===")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("\n=== 训练循环 ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    print(f"Epoch {epoch+1}/{epochs}")
    print("-" * 10)
    
    running_loss = 0.0
    print(f"训练中... (模拟)")
    
    print(f"Loss: {running_loss:.4f}")

print("\n=== 评估模型 ===")
model.eval()
print("评估中... (模拟)")
print("准确率: 95% (模拟)")

print("\n=== 特征提取模式 ===")
model = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(model.children())[:-1])
print(f"特征提取器:\n{feature_extractor}")