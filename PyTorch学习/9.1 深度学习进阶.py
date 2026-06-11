# PyTorch深度学习进阶学习
# 主要内容：自定义数据集、数据增强、迁移学习、模型集成

# 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 自定义数据集
print("=== 自定义数据集 ===")

class CustomDataset(Dataset):
    """自定义数据集类"""
    
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y

# 创建示例数据
np.random.seed(42)
X = np.random.randn(1000, 3, 32, 32).astype(np.float32)  # 1000个32x32的RGB图像
y = np.random.randint(0, 10, size=1000)  # 10个类别

print(f"数据集形状: X={X.shape}, y={y.shape}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 数据增强
print("\n=== 数据增强 ===")

# 定义数据增强
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 应用数据增强
train_dataset_aug = CustomDataset(X_train, y_train, transform=train_transform)
test_dataset_aug = CustomDataset(X_test, y_test, transform=test_transform)

train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
test_loader_aug = DataLoader(test_dataset_aug, batch_size=batch_size)

print("数据增强配置完成")

# 自定义神经网络
print("\n=== 自定义神经网络 ===")

class CNNModel(nn.Module):
    """自定义卷积神经网络"""
    
    def __init__(self, num_classes=10):
        super(CNNModel, self).__init__()
        
        # 特征提取部分
        self.features = nn.Sequential(
            # 卷积层1
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 卷积层2
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 卷积层3
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 创建模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNModel(num_classes=10).to(device)
print(f"模型设备: {device}")
print(f"参数数量: {sum(p.numel() for p in model.parameters())}")

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("模型创建完成")

# 训练函数
print("\n=== 训练模型 ===")

def train(model, train_loader, criterion, optimizer, num_epochs=5):
    """训练模型"""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
        
        # 更新学习率
        scheduler.step()
    
    return model

# 评估函数
def evaluate(model, test_loader):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total * 100
    return accuracy

# 训练模型（使用简化版本进行快速演示）
print("开始训练...")
model = train(model, train_loader_aug, criterion, optimizer, num_epochs=3)
test_acc = evaluate(model, test_loader_aug)
print(f"测试集准确率: {test_acc:.2f}%")

# 迁移学习
print("\n=== 迁移学习 ===")

# 加载预训练模型
print("加载预训练的ResNet18...")
try:
    pretrained_model = models.resnet18(pretrained=True)
    
    # 冻结参数
    for param in pretrained_model.parameters():
        param.requires_grad = False
    
    # 修改最后一层
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, 10)
    
    # 移动到设备
    pretrained_model = pretrained_model.to(device)
    
    # 只训练最后一层
    optimizer_ft = optim.Adam(pretrained_model.fc.parameters(), lr=0.001)
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pretrained_model.parameters())
    
    print(f"ResNet18加载完成")
    print(f"可训练参数: {trainable_params:,}")
    print(f"总参数: {total_params:,}")
    print(f"可训练比例: {trainable_params / total_params * 100:.2f}%")
    
    # 微调（解冻部分层）
    print("\n微调：解冻最后几层...")
    for param in pretrained_model.layer4.parameters():
        param.requires_grad = True
    
    # 更新优化器
    optimizer_ft = optim.Adam([
        {'params': pretrained_model.layer4.parameters(), 'lr': 1e-4},
        {'params': pretrained_model.fc.parameters(), 'lr': 1e-3}
    ])
    
    trainable_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print(f"微调后可训练参数: {trainable_params:,}")
    
except Exception as e:
    print(f"预训练模型加载失败: {e}")
    print("使用自定义模型演示其他功能")

# 学习率调度器
print("\n=== 学习率调度器 ===")

# 创建学习率调度器示例
lr_schedulers = {
    'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1),
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10),
    'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
}

# 可视化学习率变化
print("学习率调度器类型:")
for name, scheduler in lr_schedulers.items():
    print(f"  - {name}")

# 绘制学习率曲线
lr_values = []
lr_values_schedule = lr_schedulers['StepLR']
for _ in range(20):
    lr_values.append(optimizer.param_groups[0]['lr'])
    lr_values_schedule.step()

plt.figure(figsize=(10, 4))
plt.plot(lr_values)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('StepLR 学习率变化')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 模型保存与加载
print("\n=== 模型保存与加载 ===")

# 保存检查点
checkpoint = {
    'epoch': 5,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': 0.123,
}

torch.save(checkpoint, 'model_checkpoint.pth')
print("模型检查点已保存")

# 加载检查点
checkpoint = torch.load('model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
print(f"检查点加载完成，Epoch: {epoch}")

# 清理
import os
if os.path.exists('model_checkpoint.pth'):
    os.remove('model_checkpoint.pth')
    print("已删除临时模型文件")

# 模型集成
print("\n=== 模型集成 ===")

# 创建多个模型进行集成
num_models = 3
models = []

for i in range(num_models):
    model_ens = CNNModel(num_classes=10).to(device)
    optimizer_ens = optim.Adam(model_ens.parameters(), lr=0.001)
    
    # 简化：使用相同的初始化（实际中应该有不同的初始化）
    models.append(model_ens)
    print(f"模型 {i+1} 创建完成")

# 简单的投票集成
def ensemble_predict(models, inputs):
    """使用投票法进行集成预测"""
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())
    
    # 投票
    predictions = np.array(predictions)
    final_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=predictions
    )
    
    return final_pred

# 加权平均集成
def weighted_ensemble_predict(models, weights, inputs):
    """使用加权平均进行集成预测"""
    predictions = []
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    # 加权平均
    weighted_pred = np.sum(np.array(predictions) * np.array(weights)[:, None, None], axis=0)
    return np.argmax(weighted_pred, axis=1)

print("模型集成方法定义完成")
print("  - 简单投票法")
print("  - 加权平均法")

# 自定义损失函数
print("\n=== 自定义损失函数 ===")

class FocalLoss(nn.Module):
    """Focal Loss - 用于解决类别不平衡问题"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失 - 防止过拟合"""
    
    def __init__(self, num_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        
        # 创建平滑标签
        targets_onehot = torch.zeros_like(log_probs)
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)
        targets_onehot = targets_onehot * confidence + (1 - targets_onehot) * self.smoothing / (self.num_classes - 1)
        
        loss = (-targets_onehot * log_probs).sum(-1)
        return loss.mean()

# 创建损失函数
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
smooth_loss = LabelSmoothingLoss(num_classes=10, smoothing=0.1)

print("自定义损失函数定义完成")
print("  - Focal Loss: 处理类别不平衡")
print("  - Label Smoothing: 防止过拟合")

# 正则化技术
print("\n=== 正则化技术 ===")

# L1/L2正则化
def l1_regularization(model, lambda_l1=0.001):
    """L1正则化"""
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def l2_regularization(model, lambda_l2=0.001):
    """L2正则化"""
    l2_loss = 0
    for param in model.parameters():
        l2_loss += torch.sum(param ** 2)
    return lambda_l2 * l2_loss

# 在训练中加入正则化
def train_with_regularization(model, train_loader, criterion, optimizer, num_epochs=5, lambda_l2=0.001):
    """带正则化的训练"""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 添加L2正则化
            l2_loss = l2_regularization(model, lambda_l2)
            total_loss = loss + l2_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
    
    return model

print("正则化技术定义完成")
print("  - L1正则化")
print("  - L2正则化")
print("  - Dropout (已在模型中使用)")

# 梯度裁剪
print("\n=== 梯度裁剪 ===")

def train_with_gradient_clipping(model, train_loader, criterion, optimizer, num_epochs=5, max_norm=1.0):
    """带梯度裁剪的训练"""
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%")
    
    return model

print(f"梯度裁剪已定义，max_norm=1.0")

# 训练可视化
print("\n=== 训练可视化 ===")

# 记录训练过程
train_losses = []
train_accs = []

# 简化的训练循环（用于演示）
model_vis = CNNModel(num_classes=10).to(device)
optimizer_vis = optim.Adam(model_vis.parameters(), lr=0.001)

for epoch in range(10):
    model_vis.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader_aug):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model_vis(inputs)
        loss = criterion(outputs, labels)
        
        optimizer_vis.zero_grad()
        loss.backward()
        optimizer_vis.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # 只训练前5个batch用于演示
        if batch_idx >= 5:
            break
    
    avg_loss = epoch_loss / min(batch_idx + 1, 5)
    avg_acc = correct / total * 100
    train_losses.append(avg_loss)
    train_accs.append(avg_acc)

# 可视化训练曲线
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(train_losses, label='Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('训练损失')
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label='Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('训练准确率')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 模型分析
print("\n=== 模型分析 ===")

# 计算模型大小
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"模型参数大小: {param_size / 1024 / 1024:.2f} MB")

# 计算推理速度
input_sample = torch.randn(1, 3, 32, 32).to(device)

# 热身
for _ in range(10):
    with torch.no_grad():
        _ = model(input_sample)

# 计时
import time
num_iterations = 100
start_time = time.time()
for _ in range(num_iterations):
    with torch.no_grad():
        _ = model(input_sample)
end_time = time.time()

avg_time = (end_time - start_time) / num_iterations * 1000
print(f"平均推理时间: {avg_time:.2f} ms")
print(f"推理速度: {1000 / avg_time:.2f} FPS")

# 总结
print("\n=== PyTorch进阶学习总结 ===")
print("1. 自定义数据集和数据加载器")
print("2. 数据增强技术")
print("3. 自定义神经网络架构")
print("4. 迁移学习和模型微调")
print("5. 多种学习率调度器")
print("6. 模型保存与加载")
print("7. 模型集成方法")
print("8. 自定义损失函数")
print("9. 正则化技术")
print("10. 梯度裁剪")
print("11. 训练可视化")
print("12. 模型性能分析")

print("\nPyTorch深度学习进阶学习完成！")
