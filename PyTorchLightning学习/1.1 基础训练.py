# PyTorch Lightning基础训练学习
# 主要内容：LightningModule、Trainer、数据模块

# 导入必要的库
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 定义LightningModule
print("=== 定义LightningModule ===")

class MNISTClassifier(pl.LightningModule):
    """MNIST手写数字分类器"""

    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数

        # 定义神经网络
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # 学习率
        self.learning_rate = learning_rate

    def forward(self, x):
        """前向传播"""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # 记录日志
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # 记录验证指标
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        # 记录测试指标
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

print("LightningModule定义完成")

# 定义数据模块
print("\n=== 定义数据模块 ===")

class MNISTDataModule(pl.LightningDataModule):
    """MNIST数据模块"""

    def __init__(self, data_dir='./data', batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        """下载数据（只执行一次）"""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """设置数据（每个GPU执行一次）"""
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        """测试数据加载器"""
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

print("DataModule定义完成")

# 创建模型和数据模块
print("\n=== 创建模型和数据模块 ===")

# 创建数据模块
dm = MNISTDataModule(batch_size=64)

# 创建模型
model = MNISTClassifier(learning_rate=1e-3)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 创建训练器
print("\n=== 创建训练器 ===")

# 回调函数：保存最佳模型
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    dirpath='lightning_logs/checkpoints/',
    filename='mnist-best'
)

# 日志记录器
logger = TensorBoardLogger('lightning_logs', name='mnist_classifier')

# 创建训练器
trainer = pl.Trainer(
    max_epochs=5,
    accelerator='auto',  # 自动选择GPU或CPU
    devices='auto',      # 使用所有可用设备
    callbacks=[checkpoint_callback],
    logger=logger,
    deterministic=True
)

print(f"训练器配置: {trainer}")

# 训练模型
print("\n=== 训练模型 ===")

trainer.fit(model, datamodule=dm)

# 测试模型
print("\n=== 测试模型 ===")

trainer.test(model, datamodule=dm, ckpt_path='best')

# 加载最佳模型
print("\n=== 加载最佳模型 ===")

best_model = MNISTClassifier.load_from_checkpoint(
    checkpoint_callback.best_model_path
)
best_model.eval()

print(f"最佳模型路径: {checkpoint_callback.best_model_path}")

# 进行预测
print("\n=== 进行预测 ===")

# 获取测试数据
test_data = datasets.MNIST('./data', train=False, transform=dm.transform)
test_loader = DataLoader(test_data, batch_size=1)

# 进行预测
for x, y in test_loader:
    with torch.no_grad():
        logits = best_model(x)
        pred = torch.argmax(logits, dim=1).item()
    print(f"真实标签: {y.item()}, 预测: {pred}")
    break

# 清理检查点目录
import os
import shutil
ckpt_dir = 'lightning_logs/checkpoints/'
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
    print(f"\n已删除检查点目录: {ckpt_dir}")

print("\nPyTorch Lightning基础训练学习完成！")