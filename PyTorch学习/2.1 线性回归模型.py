# PyTorch线性回归模型学习
# 主要内容：使用PyTorch实现简单线性回归，包括数据集创建、模型定义、训练循环

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("=== 创建模拟数据集 ===")

# 创建模拟数据
torch.manual_seed(42)
x = torch.linspace(0, 10, 100).unsqueeze(1)
y = 2 * x + 1 + torch.randn(100, 1) * 0.5

plt.scatter(x.numpy(), y.numpy(), label='数据点')
plt.title('模拟线性数据')
plt.legend()
plt.show()

print("\n=== 定义线性回归模型 ===")

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
print(f"模型参数: {list(model.parameters())}")

print("\n=== 训练模型 ===")

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
loss_history = []

for epoch in range(epochs):
    model.train()
    
    # 前向传播
    y_pred = model(x)
    
    # 计算损失
    loss = criterion(y_pred, y)
    loss_history.append(loss.item())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("\n=== 训练结果 ===")
print(f"学习到的权重: {model.linear.weight.item():.4f}")
print(f"学习到的偏置: {model.linear.bias.item():.4f}")

plt.plot(loss_history)
plt.title('训练损失变化')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 预测
model.eval()
with torch.no_grad():
    y_pred = model(x)
    plt.scatter(x.numpy(), y.numpy(), label='数据点')
    plt.plot(x.numpy(), y_pred.numpy(), 'r-', label='拟合直线')
    plt.legend()
    plt.title('线性回归拟合结果')
    plt.show()