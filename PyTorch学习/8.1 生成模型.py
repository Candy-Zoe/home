# PyTorch生成模型学习
# 主要内容：VAE、GAN、自编码器、变分推断

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

print("=== 加载数据 ===")
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
print(f"训练数据量: {len(train_data)}")

print("\n=== 自编码器 ===")
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

autoencoder = Autoencoder()
print("自编码器已创建")

print("\n=== 变分自编码器(VAE) ===")
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28*28),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

vae = VAE()
print("VAE已创建")

print("\n=== VAE损失函数 ===")
def vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

print("VAE损失函数已定义")

print("\n=== GAN生成器 ===")
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

print("GAN生成器已创建")

print("\n=== GAN判别器 ===")
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

print("GAN判别器已创建")

print("\n=== 训练自编码器 ===")
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(3):
    total_loss = 0
    for data, _ in train_loader:
        data = data.view(-1, 28*28)
        
        optimizer.zero_grad()
        output = autoencoder(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print("\n=== 可视化重构 ===")
data, _ = next(iter(train_loader))
data = data.view(-1, 28*28)
recon = autoencoder(data).detach()

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(5):
    axes[0, i].imshow(data[i].view(28, 28), cmap='gray')
    axes[0, i].set_title('原始')
    axes[1, i].imshow(recon[i].view(28, 28), cmap='gray')
    axes[1, i].set_title('重构')
plt.tight_layout()
plt.show()

print("\n=== 训练VAE ===")
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

for epoch in range(3):
    total_loss = 0
    for data, _ in train_loader:
        data = data.view(-1, 28*28)
        
        optimizer_vae.zero_grad()
        recon, mu, logvar = vae(data)
        loss = vae_loss(recon, data, mu, logvar)
        loss.backward()
        optimizer_vae.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

print("\n=== 从VAE生成新样本 ===")
with torch.no_grad():
    z = torch.randn(5, 20)
    samples = vae.decode(z).view(-1, 28, 28)

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    axes[i].imshow(samples[i], cmap='gray')
    axes[i].set_title(f'样本{i+1}')
plt.tight_layout()
plt.show()

print("\n=== 清理 ===")
import shutil
import os
if os.path.exists('./data'):
    shutil.rmtree('./data')
    print("数据目录已删除")