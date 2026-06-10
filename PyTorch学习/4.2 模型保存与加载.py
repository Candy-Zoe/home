# PyTorch模型保存与加载学习
# 主要内容：模型保存、模型加载、断点续训

import torch
import torch.nn as nn
import torch.optim as optim

print("=== 定义简单模型 ===")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"原始模型参数:\n{list(model.parameters())[0][0]}")

print("\n=== 保存模型 ===")

torch.save(model.state_dict(), 'model_weights.pth')

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 10,
    'loss': 0.5
}, 'model_checkpoint.pth')

print("模型已保存")

print("\n=== 加载模型 ===")
new_model = SimpleModel()
new_model.load_state_dict(torch.load('model_weights.pth'))
print(f"加载后模型参数:\n{list(new_model.parameters())[0][0]}")

print("\n=== 加载检查点 ===")
checkpoint = torch.load('model_checkpoint.pth')
new_model2 = SimpleModel()
new_optimizer = optim.Adam(new_model2.parameters())

new_model2.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"加载的epoch: {epoch}")
print(f"加载的loss: {loss}")

print("\n=== 切换设备 ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

sample = torch.randn(1, 10).to(device)
output = model(sample)
print(f"设备: {device}")
print(f"输出形状: {output.shape}")

print("\n=== 清理测试文件 ===")
import os
for f in ['model_weights.pth', 'model_checkpoint.pth']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")