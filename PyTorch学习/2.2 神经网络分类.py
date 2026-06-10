# PyTorch神经网络分类学习
# 主要内容：使用PyTorch实现简单的全连接神经网络进行手写数字分类

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

print("=== 加载MNIST数据集 ===")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

print("\n=== 定义神经网络模型 ===")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net()
print(model)

print("\n=== 训练模型 ===")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
train_loss_history = []
test_acc_history = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_loss_history.append(train_loss)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_acc = correct / len(test_loader.dataset)
    test_acc_history.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Test Acc: {test_acc:.4f}")

print("\n=== 可视化结果 ===")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history)
plt.title('训练损失')
plt.xlabel('Epoch')

plt.subplot(1, 2, 2)
plt.plot(test_acc_history)
plt.title('测试准确率')
plt.xlabel('Epoch')
plt.show()

print("\n=== 测试单个图片 ===")
model.eval()
with torch.no_grad():
    sample = test_dataset[0][0].unsqueeze(0)
    output = model(sample)
    pred = output.argmax(dim=1).item()
    print(f"预测结果: {pred}")
    print(f"真实标签: {test_dataset[0][1]}")
    
    plt.imshow(test_dataset[0][0].squeeze().numpy(), cmap='gray')
    plt.title(f"预测: {pred}, 真实: {test_dataset[0][1]}")
    plt.show()