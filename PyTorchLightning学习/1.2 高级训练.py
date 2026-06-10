# PyTorch Lightning高级训练学习
# 主要内容：回调函数、日志记录、早停机制

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

print("=== 定义模型 ===")
class MNISTModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        return [optimizer], [scheduler]

print("\n=== 准备数据 ===")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_dataset, val_dataset = random_split(dataset, [55000, 5000])

train_loader = DataLoader(train_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

print("\n=== 创建回调函数 ===")
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max')

print("\n=== 创建Trainer ===")
trainer = pl.Trainer(
    max_epochs=5,
    accelerator='auto',
    devices='auto',
    callbacks=[early_stopping, checkpoint_callback],
    logger=False,
    enable_checkpointing=True
)

print("\n=== 训练模型 ===")
model = MNISTModel(lr=1e-3)
trainer.fit(model, train_loader, val_loader)

print("\n=== 清理数据 ===")
import shutil
import os
if os.path.exists('./data'):
    shutil.rmtree('./data')
if os.path.exists('./lightning_logs'):
    shutil.rmtree('./lightning_logs')
print("已删除数据和日志目录")