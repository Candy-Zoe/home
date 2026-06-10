# PyTorch Lightning基础训练学习
# 主要内容：LightningModule、Trainer、数据加载

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

print("=== 定义LightningModule ===")
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

print("\n=== 准备数据 ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

print("\n=== 创建Trainer ===")
trainer = pl.Trainer(
    max_epochs=3,
    accelerator='auto',
    devices='auto',
    logger=False,
    enable_checkpointing=False
)

print("\n=== 训练模型 ===")
model = MNISTModel()
trainer.fit(model, train_loader, val_loader)

print("\n=== 测试模型 ===")
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32)
trainer.test(model, test_loader)

print("\n=== 清理数据 ===")
import shutil
import os
if os.path.exists('./data'):
    shutil.rmtree('./data')
    print("已删除数据目录")