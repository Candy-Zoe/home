# PyTorch线性回归模型学习
# 主要内容：线性回归模型构建、训练过程、模型评估

# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
print("=== 生成示例数据 ===")

# 设置随机种子以保证结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 生成100个样本
n_samples = 100

# 真实的线性关系：y = 2x + 1 + noise
X = torch.randn(n_samples, 1)  # 输入特征
noise = torch.randn(n_samples, 1) * 0.5  # 噪声
y = 2 * X + 1 + noise  # 目标值

print(f"输入数据形状: {X.shape}")
print(f"目标数据形状: {y.shape}")
print(f"X范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"y范围: [{y.min():.2f}, {y.max():.2f}]")

# 定义线性回归模型
print("\n=== 定义线性回归模型 ===")

class LinearRegression(nn.Module):
    """线性回归模型类"""

    def __init__(self, input_dim, output_dim):
        """初始化模型
        参数:
            input_dim: 输入特征维度
            output_dim: 输出维度
        """
        super(LinearRegression, self).__init__()
        # 定义线性层（全连接层）
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """前向传播"""
        return self.linear(x)

# 创建模型实例
model = LinearRegression(input_dim=1, output_dim=1)
print(model)

# 查看模型参数
print("\n模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}, 值 = {param.item():.4f}")

# 定义损失函数和优化器
print("\n=== 定义损失函数和优化器 ===")

# 均方误差损失函数
criterion = nn.MSELoss()

# 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# 准备数据
print("\n=== 准备数据 ===")

# 创建数据集
dataset = TensorDataset(X, y)

# 创建数据加载器（用于批量训练）
batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"数据集大小: {len(dataset)}")
print(f"批量大小: {batch_size}")
print(f"批次数: {len(dataloader)}")

# 训练模型
print("\n=== 训练模型 ===")

# 设置训练轮数
n_epochs = 100

# 用于记录训练过程
losses = []

# 设置为训练模式
model.train()

for epoch in range(n_epochs):
    epoch_loss = 0.0

    for batch_X, batch_y in dataloader:
        # 1. 前向传播
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # 2. 反向传播
        optimizer.zero_grad()  # 清零梯度
        loss.backward()

        # 3. 更新参数
        optimizer.step()

        # 累积损失
        epoch_loss += loss.item()

    # 计算平均损失
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

# 训练完成后的模型参数
print("\n训练后的模型参数:")
for name, param in model.named_parameters():
    print(f"  {name}: 值 = {param.item():.4f}")

# 理论参数应该是：weight ≈ 2, bias ≈ 1
print(f"\n理论参数: weight = 2.0, bias = 1.0")
print(f"学习到的参数: weight = {model.linear.weight.item():.4f}, bias = {model.linear.bias.item():.4f}")

# 评估模型
print("\n=== 评估模型 ===")

# 设置为评估模式
model.eval()

# 计算整个数据集上的损失
with torch.no_grad():
    y_pred = model(X)
    total_loss = criterion(y_pred, y)
    print(f"整个数据集上的MSE: {total_loss.item():.4f}")

# 计算R²分数
def r2_score(y_true, y_pred):
    """计算R²分数"""
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot

r2 = r2_score(y, y_pred)
print(f"R²分数: {r2:.4f}")

# 可视化结果
print("\n=== 可视化结果 ===")

# 绘制训练损失曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.grid(True)

# 绘制数据点和拟合线
plt.subplot(1, 2, 2)
plt.scatter(X.numpy(), y.numpy(), alpha=0.5, label='数据点')

# 生成拟合线
X_line = torch.linspace(X.min(), X.max(), 100).reshape(-1, 1)
with torch.no_grad():
    y_line = model(X_line)

plt.plot(X_line.numpy(), y_line.numpy(), color='red', linewidth=2, label='拟合线')
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'线性回归拟合 (R²={r2:.4f})')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 保存和加载模型
print("\n=== 保存和加载模型 ===")

# 方法1：保存整个模型
torch.save(model, 'linear_regression_model.pth')
print("模型已保存为 linear_regression_model.pth")

# 方法2：只保存模型参数（推荐）
torch.save(model.state_dict(), 'linear_regression_params.pth')
print("模型参数已保存为 linear_regression_params.pth")

# 加载模型
loaded_model = LinearRegression(1, 1)
loaded_model.load_state_dict(torch.load('linear_regression_params.pth'))
loaded_model.eval()

print("\n加载模型验证:")
with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    test_output = loaded_model(test_input)
    print(f"输入: {test_input.item()}, 预测输出: {test_output.item():.4f}")