# PyTorch Lightning高级训练技术学习
# 主要内容：多任务学习、GAN、对比学习、分布式训练进阶

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

print("=== 多任务学习模型 ===")
class MultiTaskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        self.task_a_head = nn.Linear(32, 10)
        self.task_b_head = nn.Linear(32, 2)
    
    def forward(self, x):
        features = self.encoder(x)
        out_a = self.task_a_head(features)
        out_b = self.task_b_head(features)
        return out_a, out_b
    
    def training_step(self, batch, batch_idx):
        x, y_a, y_b = batch
        pred_a, pred_b = self(x)
        
        loss_a = F.cross_entropy(pred_a, y_a)
        loss_b = F.cross_entropy(pred_b, y_b)
        loss = loss_a + 0.5 * loss_b
        
        self.log('train_loss_a', loss_a)
        self.log('train_loss_b', loss_b)
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

print("多任务学习模型已定义")

print("\n=== GAN模型 ===")
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x.view(-1, 784))

print("GAN模型已定义")

print("\n=== 自定义回调 ===")
class ImageLoggerCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, trainer, pl_module):
        print(f"Epoch {trainer.current_epoch} 完成")

print("自定义回调已定义")

print("\n=== 分布式训练 ===")
print("使用DistributedDataParallel进行多GPU训练")

print("\n=== 混合精度训练 ===")
trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    precision=16,
    max_epochs=10
)
print("混合精度训练配置已创建")

print("\n=== 学习率调度器 ===")
class LRSchedulerModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.cross_entropy(self.net(x), y)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

print("学习率调度模块已定义")

print("\n=== Early Stopping ===")
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

early_stop = EarlyStopping(
    monitor='train_loss',
    patience=5,
    mode='min'
)
print("Early Stopping回调已创建")

print("\n=== 模型检查点 ===")
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    dirpath='./checkpoints',
    filename='{epoch}-{train_loss:.2f}',
    save_top_k=3,
    monitor='train_loss',
    mode='min'
)
print("模型检查点回调已创建")

print("\n=== 日志记录 ===")
class LoggingModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', (y_pred.argmax(1) == y).float().mean(), prog_bar=True)
        
        return loss
    
    def configure_metrics(self):
        return {'acc': pl.metrics.Accuracy()}

print("日志记录模块已定义")