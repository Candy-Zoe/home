# PyTorch数据加载与预处理学习
# 主要内容：Dataset、DataLoader、数据变换

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

print("=== 自定义Dataset ===")

class CustomDataset(Dataset):
    def __init__(self, transform=None):
        self.data = np.random.randn(100, 28, 28)
        self.labels = np.random.randint(0, 10, 100)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

dataset = CustomDataset()
print(f"数据集大小: {len(dataset)}")
sample, label = dataset[0]
print(f"样本形状: {sample.shape}, 标签: {label}")

print("\n=== 数据变换 ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomDataset(transform=transform)
sample, label = dataset[0]
print(f"变换后样本形状: {sample.shape}")
print(f"样本均值: {sample.mean():.4f}, 标准差: {sample.std():.4f}")

print("\n=== DataLoader ===")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for batch_idx, (data, labels) in enumerate(dataloader):
    print(f"批次 {batch_idx}:")
    print(f"  数据形状: {data.shape}")
    print(f"  标签: {labels}")
    if batch_idx == 2:
        break

print("\n=== 内置数据集 ===")
mnist_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
mnist_loader = DataLoader(mnist_dataset, batch_size=64, shuffle=True)

data_iter = iter(mnist_loader)
images, labels = next(data_iter)
print(f"MNIST批次形状: {images.shape}")

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i].squeeze().numpy(), cmap='gray')
    plt.title(f"标签: {labels[i].item()}")
    plt.axis('off')
plt.show()