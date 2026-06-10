# Weights & Biases实验追踪进阶学习
# 主要内容：多实验对比、超参数调优、模型版本管理、可视化进阶

import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("=== 初始化WandB ===")
wandb.init(project="ai-learning-demo", entity=None, name="进阶教程实验")

print("\n=== 记录多种数据类型 ===")
wandb.log({
    "accuracy": 0.95,
    "loss": 0.05,
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=[0, 1, 2, 0, 1, 2],
        preds=[0, 2, 1, 0, 0, 1],
        class_names=["A", "B", "C"]
    ),
    "chart": wandb.plot.line_series(
        xs=[1, 2, 3, 4, 5],
        ys=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]],
        keys=["train", "val"],
        title="训练曲线"
    )
})
print("多种数据类型已记录")

print("\n=== 超参数扫描 ===")
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'accuracy', 'goal': 'maximize'},
    'parameters': {
        'n_estimators': {'values': [50, 100, 200]},
        'max_depth': {'values': [3, 5, 7, None]},
        'learning_rate': {'values': [0.01, 0.1, 0.3]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="ai-learning-demo")
print(f"Sweep ID: {sweep_id}")

print("\n=== Sweep训练函数 ===")
def train_with_sweep():
    wandb.init()
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = GradientBoostingClassifier(
        n_estimators=wandb.config.n_estimators,
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    wandb.log({"accuracy": accuracy})

print("Sweep训练函数已定义")

print("\n=== 记录图表 ===")
x = np.arange(0, 100)
y = np.cumsum(np.random.randn(100))

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.title('训练曲线')
plt.xlabel('Step')
plt.ylabel('Value')
plt.savefig('temp_plot.png')
plt.close()

wandb.log({"custom_plot": wandb.Image('temp_plot.png')})
print("自定义图表已记录")

print("\n=== 记录视频 ===")
print("视频记录示例（需要实际视频数据）")

print("\n=== 分组实验 ===")
wandb.init(group="baseline", job_type="train")
wandb.log({"metric": 0.5, "split": "train"})

wandb.init(group="baseline", job_type="eval")
wandb.log({"metric": 0.6, "split": "val"})
print("分组实验已记录")

print("\n=== 保存模型 ===")
model = RandomForestClassifier()
model.fit([[0, 1], [1, 0]], [0, 1])
wandb.sklearn.plot_classifier(
    model,
    [[0, 1], [1, 0]],
    [0, 1],
    [0.6, 0.4],
    [0, 1],
    ["class_0", "class_1"]
)
print("模型分析图表已记录")

print("\n=== 查看运行历史 ===")
api = wandb.Api()
runs = api.runs("ai-learning-demo")
print(f"项目运行数: {len(runs)}")

print("\n=== 比较实验 ===")
print("使用WandB UI比较不同实验的超参数和性能")

print("\n=== 清理 ===")
import os
if os.path.exists('temp_plot.png'):
    os.remove('temp_plot.png')
    print("临时文件已删除")

wandb.finish()
print("WandB会话已结束")