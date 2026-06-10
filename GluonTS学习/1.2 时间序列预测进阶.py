# GluonTS深度时间序列预测进阶学习
# 主要内容：DeepAR进阶、Transformer模型、多步预测、概率预测

import mxnet as mx
from gluonts.dataset.repository import get_dataset, dataset_recipes
from gluonts.trainer import Trainer
from gluonts.model.deepar import DeepAREstimator
from gluonts.modelTransformer import TransformerEstimator
from gluonts.model.n_beats import NBeatsEstimator
from gluonts.evaluation import Evaluator
from gluonts.evaluation import make_evaluation_predictions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print("=== GluonTS可用数据集 ===")
print("可用数据集列表:")
for name in list(dataset_recipes.keys())[:5]:
    print(f"  - {name}")

print("\n=== 加载数据集 ===")
dataset = get_dataset("airline")
print(f"训练集长度: {len(dataset.train)}")
print(f"测试集长度: {len(dataset.test)}")

print("\n=== 获取数据条目 ===")
entry = next(iter(dataset.train))
print(f"数据字段: {entry.keys()}")
print(f"开始时间: {entry['start']}")
print(f"目标值长度: {len(entry['target'])}")

print("\n=== DeepAR模型 ===")
estimator = DeepAREstimator(
    freq="MS",
    prediction_length=12,
    context_length=24,
    trainer=Trainer(epochs=5, learning_rate=1e-3, ctx=mx.cpu()),
    num_layers=2,
    hidden_size=32
)

print("DeepAR估计器已创建")

print("\n=== Transformer模型 ===")
transformer_estimator = TransformerEstimator(
    freq="MS",
    prediction_length=12,
    context_length=24,
    trainer=Trainer(epochs=5, learning_rate=1e-3, ctx=mx.cpu()),
    d_model=32,
    num_heads=4,
    num_layers=2
)

print("Transformer估计器已创建")

print("\n=== N-BEATS模型 ===")
nbeats_estimator = NBeatsEstimator(
    freq="MS",
    prediction_length=12,
    context_length=24,
    trainer=Trainer(epochs=5, learning_rate=1e-3, ctx=mx.cpu()),
    num_stacks=1,
    num_blocks=1,
    widths=32
)

print("N-BEATS估计器已创建")

print("\n=== 训练和预测 ===")
predictor = estimator.train(dataset.train)
print("模型训练完成")

print("\n=== 预测评估 ===")
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
    num_samples=100
)

forecasts = list(forecast_it)
tss = list(ts_it)

print(f"预测数量: {len(forecasts)}")
print(f"第一个预测的长度: {len(forecasts[0])}")

print("\n=== 预测可视化 ===")
def plot_forecast(ts, forecast, label):
    ax = ts.plot(label="实际值")
    forecast.plot(ax=ax, label=label)

plot_forecast(tss[0], forecasts[0], "预测")
plt.title('时间序列预测')
plt.legend()
plt.show()

print("\n=== 评估指标 ===")
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(dataset.test))

print("聚合指标:")
for key, value in agg_metrics.items():
    print(f"  {key}: {value:.4f}")

print("\n=== 概率预测 ===")
sample_units = 100
sample_length = 50
sample_data = np.random.randn(sample_units, sample_length).astype(np.float32)

from gluonts.dataset.array态 import ArrayDataset
sample_dataset = ArrayDataset(sample_data)

print(f"样本数据集条目数: {len(list(sample_dataset))}")

print("\n=== 自定义数据集 ===")
from gluonts.dataset.common import ListDataset

custom_data = [
    {
        "start": pd.Period("2020-01-01", freq="D"),
        "target": np.random.randn(100).astype(np.float32)
    },
    {
        "start": pd.Period("2020-01-01", freq="D"),
        "target": np.random.randn(100).astype(np.float32)
    }
]

custom_ds = ListDataset(custom_data, freq="D")
print(f"自定义数据集条目数: {len(list(custom_ds))}")