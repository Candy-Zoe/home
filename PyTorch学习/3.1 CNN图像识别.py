# PyTorch卷积神经网络图像识别学习
# 主要内容：CNN基础、LeNet模型构建、训练流程、图像分类

# 导入PyTorch库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 加载MNIST数据集
print("=== 加载MNIST数据集 ===")

from torchvision import datasets

# 下载并加载训练集
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

# 下载并加载测试集
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 查看数据形状
sample_data, sample_label = train_dataset[0]
print(f"单张图像形状: {sample_data.shape}")
print(f"标签: {sample_label}")

# 创建数据加载器
print("\n=== 创建数据加载器 ===")

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# 定义LeNet-5模型
print("\n=== 定义LeNet-5模型 ===")

class LeNet5(nn.Module):
    """LeNet-5卷积神经网络"""

    def __init__(self):
        super(LeNet5, self).__init__()

        # 第一个卷积层: 1个输入通道, 6个输出通道, 5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)

        # 第二个卷积层: 6个输入通道, 16个输出通道, 5x5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # 展平后16*5*5=400
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # 激活函数
        self.relu = nn.ReLU()

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 第一层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14

        # 第二层: 卷积 -> ReLU -> 池化
        x = self.pool(self.relu(self.conv2(x)))  # 10x10 -> 5x5

        # 展平
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 创建模型实例
model = LeNet5()
print(model)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数数量: {total_params:,}")
print(f"可训练参数数量: {trainable_params:,}")

# 定义损失函数和优化器
print("\n=== 定义损失函数和优化器 ===")

criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

print(f"损失函数: {criterion}")
print(f"优化器: Adam, 学习率=0.001")

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc

# 测试函数
def test(model, test_loader, criterion):
    """测试模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc

# 训练模型
print("\n=== 训练模型 ===")

num_epochs = 5

train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)

    # 测试
    test_loss, test_acc = test(model, test_loader, criterion)

    # 记录
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
    print(f"  测试 - 损失: {test_loss:.4f}, 准确率: {test_acc:.2f}%")

# 可视化训练过程
print("\n=== 可视化训练过程 ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
axes[0].plot(range(1, num_epochs+1), train_losses, label='训练损失')
axes[0].plot(range(1, num_epochs+1), test_losses, label='测试损失')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('损失')
axes[0].set_title('训练和测试损失')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率曲线
axes[1].plot(range(1, num_epochs+1), train_accs, label='训练准确率')
axes[1].plot(range(1, num_epochs+1), test_accs, label='测试准确率')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('准确率 (%)')
axes[1].set_title('训练和测试准确率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 模型预测
print("\n=== 模型预测示例 ===")

# 获取几个测试样本
model.eval()
samples = []
true_labels = []
predictions = []

with torch.no_grad():
    for i in range(5):
        data, target = test_dataset[i]
        output = model(data.unsqueeze(0))
        pred = output.argmax(1).item()

        samples.append(data.squeeze().numpy())
        true_labels.append(target)
        predictions.append(pred)

# 可视化预测结果
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, (img, true, pred) in enumerate(zip(samples, true_labels, predictions)):
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'真实: {true}\n预测: {pred}')
    axes[i].axis('off')

plt.suptitle('MNIST手写数字识别结果')
plt.tight_layout()
plt.show()

# 保存和加载模型
print("\n=== 保存和加载模型 ===")

# 保存整个模型
torch.save(model.state_dict(), 'lenet5_mnist.pth')
print("模型已保存为 lenet5_mnist.pth")

# 加载模型
loaded_model = LeNet5()
loaded_model.load_state_dict(torch.load('lenet5_mnist.pth'))
loaded_model.eval()

# 验证加载的模型
test_loss, test_acc = test(loaded_model, test_loader, criterion)
print(f"加载模型测试准确率: {test_acc:.2f}%")

print("\n学习完成！")