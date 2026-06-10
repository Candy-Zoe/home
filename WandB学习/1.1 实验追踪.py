# Weights & Biases实验追踪学习
# 主要内容：初始化、记录指标、超参数追踪

import wandb
import random

print("=== 初始化WandB ===")
wandb.init(project="demo-project", name="test-run", mode="offline")

print("\n=== 记录超参数 ===")
config = wandb.config
config.learning_rate = 0.01
config.batch_size = 32
config.epochs = 10
print(f"配置参数: {dict(config)}")

print("\n=== 记录指标 ===")
for epoch in range(5):
    loss = 1.0 / (epoch + 1) + random.random() * 0.1
    accuracy = 0.5 + epoch * 0.1 + random.random() * 0.05
    
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })
    
    print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")

print("\n=== 记录图像 ===")
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title('正弦曲线')
wandb.log({"plot": wandb.Image(plt)})
plt.close()

print("\n=== 记录模型 ===")
model_artifact = wandb.Artifact('model', type='model')
model_artifact.add_file('model.pt')
wandb.log_artifact(model_artifact)

print("\n=== 记录表格 ===")
data = [[i, i**2, i**3] for i in range(10)]
table = wandb.Table(data=data, columns=["x", "x²", "x³"])
wandb.log({"table": table})

print("\n=== 完成实验 ===")
wandb.finish()

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('model.pt'):
    os.remove('model.pt')
if os.path.exists('wandb'):
    import shutil
    shutil.rmtree('wandb')
print("已删除测试文件")