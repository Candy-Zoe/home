# PyTorch分布式训练学习
# 主要内容：数据并行、分布式数据并行、多GPU训练

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

print("=== 检查GPU可用性 ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"GPU数量: {torch.cuda.device_count()}")

print("\n=== 创建简单模型 ===")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("\n=== 数据并行 (DataParallel) ===")
if torch.cuda.device_count() > 1:
    model = SimpleModel()
    model = nn.DataParallel(model)
    model.to(device)
    print("已启用数据并行")
else:
    model = SimpleModel().to(device)
    print("使用单GPU训练")

print("\n=== 创建数据 ===")
X = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print("\n=== 训练循环 ===")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

print("\n=== 保存模型 ===")
if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), 'model.pth')
else:
    torch.save(model.state_dict(), 'model.pth')
print("模型已保存")

print("\n=== 加载模型 ===")
model = SimpleModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
print("模型已加载")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('model.pth'):
    os.remove('model.pth')
    print("已删除测试模型")