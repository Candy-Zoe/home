# Prophet时间序列预测学习
# 主要内容：数据准备、模型训练、预测、可视化

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
dates = pd.date_range('2020-01-01', periods=365)
trend = np.linspace(0, 10, 365)
seasonality = 5 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = np.random.randn(365) * 0.5

data = pd.DataFrame({
    'ds': dates,
    'y': trend + seasonality + noise
})

print(f"数据前5行:\n{data.head()}")

print("\n=== 创建Prophet模型 ===")
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(data)

print("\n=== 创建预测未来数据 ===")
future = model.make_future_dataframe(periods=30)
print(f"未来数据前5行:\n{future.head()}")

print("\n=== 执行预测 ===")
forecast = model.predict(future)
print(f"预测结果列: {forecast.columns}")
print(f"预测结果前5行:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()}")

print("\n=== 可视化预测 ===")
fig1 = model.plot(forecast)
plt.title('时间序列预测')
plt.show()

fig2 = model.plot_components(forecast)
plt.show()

print("\n=== 自定义季节性 ===")
model2 = Prophet(yearly_seasonality=False)
model2.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model2.fit(data)
forecast2 = model2.predict(future)
print("自定义季节性模型已训练")