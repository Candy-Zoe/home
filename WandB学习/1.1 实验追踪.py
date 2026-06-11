# Weights & Biases实验追踪学习
# 主要内容：WandB初始化、日志记录、超参数追踪

# 导入必要的库
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# 初始化WandB
print("=== 初始化WandB ===")

# 登录WandB（需要API key）
try:
    wandb.login()
    print("WandB登录成功")
except:
    print("WandB登录失败，使用离线模式")

# 创建WandB运行
wandb.init(
    project="pytorch-mnist-demo",  # 项目名称
    name="mnist-classifier",       # 运行名称
    config={                       # 配置参数
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs": 3,
        "optimizer": "Adam",
        "model_architecture": "CNN"
    },
    mode="online" if wandb.run else "offline"  # 根据登录状态选择模式
)

# 获取配置
config = wandb.config
print(f"配置参数: {dict(config)}")

# 定义模型
print("\n=== 定义模型 ===")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 记录模型架构
wandb.watch(model, log="all")

# 准备数据
print("\n=== 准备数据 ===")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_data, val_data = random_split(dataset, [55000, 5000])
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.batch_size)
test_loader = DataLoader(test_data, batch_size=config.batch_size)

print(f"训练集大小: {len(train_data)}")
print(f"验证集大小: {len(val_data)}")
print(f"测试集大小: {len(test_data)}")

# 训练模型
print("\n=== 训练模型 ===")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# 训练循环
for epoch in range(config.epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += data.size(0)

        # 记录每批的损失
        wandb.log({
            "batch_loss": loss.item(),
            "batch_acc": pred.eq(target.view_as(pred)).float().mean().item()
        })

    # 计算训练指标
    train_loss /= train_total
    train_acc = train_correct / train_total

    # 验证
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            val_correct += pred.eq(target.view_as(pred)).sum().item()
            val_total += data.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total

    # 记录epoch指标
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc
    })

    print(f"Epoch {epoch+1}/{config.epochs}")
    print(f"  训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
    print(f"  验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")

# 测试模型
print("\n=== 测试模型 ===")

model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += criterion(output, target).item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()
        test_total += data.size(0)

test_loss /= test_total
test_acc = test_correct / test_total

print(f"测试损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

# 记录测试结果
wandb.log({
    "test_loss": test_loss,
    "test_acc": test_acc
})

# 记录图像示例
print("\n=== 记录图像示例 ===")

# 获取一些测试图像
images, labels = next(iter(test_loader))
predictions = model(images).argmax(dim=1)

# 创建图像网格
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f"真实: {labels[i].item()}\n预测: {predictions[i].item()}")
    ax.axis('off')

# 记录图像到WandB
wandb.log({"test_predictions": fig})
plt.close()

# 记录混淆矩阵
print("\n=== 记录混淆矩阵 ===")

from sklearn.metrics import confusion_matrix
import seaborn as sns

# 获取所有预测结果
all_preds = []
all_targets = []

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        all_preds.extend(output.argmax(dim=1).tolist())
        all_targets.extend(target.tolist())

# 计算混淆矩阵
cm = confusion_matrix(all_targets, all_preds)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('预测')
plt.ylabel('真实')
plt.title('混淆矩阵')

# 记录混淆矩阵到WandB
wandb.log({"confusion_matrix": plt})
plt.close()

# 记录模型参数直方图
print("\n=== 记录参数直方图 ===")

# 记录权重直方图
for name, param in model.named_parameters():
    if 'weight' in name:
        wandb.log({f"weights/{name}": wandb.Histogram(param.detach().cpu().numpy())})

# 记录配置更新
print("\n=== 更新配置 ===")

wandb.config.update({
    "final_test_acc": test_acc,
    "total_epochs_trained": config.epochs
})

# 保存模型
print("\n=== 保存模型 ===")

torch.save(model.state_dict(), 'mnist_model.pth')
wandb.save('mnist_model.pth')
print("模型已保存并上传到WandB")

# 结束WandB运行
wandb.finish()
print("\nWandB运行已结束")

# 清理本地模型文件
import os
if os.path.exists('mnist_model.pth'):
    os.remove('mnist_model.pth')
    print("本地模型文件已删除")

print("\nWandB实验追踪学习完成！")