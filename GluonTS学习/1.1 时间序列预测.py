# GluonTS时间序列预测学习
# 主要内容：数据准备、模型训练、预测、评估

from gluonts.dataset.util import to_pandas
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
import matplotlib.pyplot as plt

print("=== 加载数据集 ===")
dataset = get_dataset("m4_hourly", regenerate=False)
print(f"数据集: {dataset}")

print("\n=== 查看数据 ===")
train_entry = next(iter(dataset.train))
train_series = to_pandas(train_entry)
print(f"时间序列长度: {len(train_series)}")
print(f"时间序列前5个值:\n{train_series.head()}")

print("\n=== 可视化时间序列 ===")
train_series.plot()
plt.title('时间序列示例')
plt.show()

print("\n=== 创建模型 ===")
estimator = SimpleFeedForwardEstimator(
    prediction_length=24,
    context_length=48,
    trainer=Trainer(epochs=5)
)

print("\n=== 训练模型 ===")
predictor = estimator.train(training_data=dataset.train)

print("\n=== 生成预测 ===")
test_entry = next(iter(dataset.test))
test_series = to_pandas(test_entry)
forecast = predictor.predict(test_entry)

print("\n=== 可视化预测 ===")
fig, ax = plt.subplots(figsize=(10, 5))
test_series.plot(ax=ax, label='真实值')
forecast.plot(ax=ax, label='预测值')
plt.title('时间序列预测')
plt.legend()
plt.show()

print("\n=== 其他模型 ===")
from gluonts.model.deepar import DeepAREstimator

deepar_estimator = DeepAREstimator(
    prediction_length=24,
    freq='H',
    trainer=Trainer(epochs=3)
)
print("DeepAR模型已创建")