# Prophet时间序列预测进阶学习
# 主要内容：多季节性、节假日效应、趋势changepoint、交叉验证、性能优化

from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=730, freq='D')
df = pd.DataFrame({
    'ds': dates,
    'y': 100 + 20*np.sin(np.arange(730) * 2 * np.pi / 365) + np.random.randn(730) * 5
})

print(f"数据形状: {df.shape}")
print(df.head())

print("\n=== 多季节性建模 ===")
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative'
)
m.fit(df)

future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.title('Prophet预测结果')
plt.show()

fig2 = m.plot_components(forecast)
plt.show()

print("\n=== 添加节假日效应 ===")
holidays = pd.DataFrame({
    'holiday': 'conference',
    'ds': pd.to_datetime(['2020-03-15', '2020-06-20', '2020-09-10']),
    'lower_window': 0,
    'upper_window': 1,
})

m_holidays = Prophet(holidays=holidays)
m_holidays.fit(df)

forecast_holidays = m_holidays.predict(future)
print("节假日效应已添加")

print("\n=== 自定义changepoints ===")
m_custom = Prophet(
    changepoint_prior_scale=0.5,
    n_changepoints=25,
    changepoint_range=0.8
)
m_custom.fit(df)

forecast_custom = m_custom.predict(future)
fig = m_custom.plot(forecast_custom)
plt.title('自定义changepoints预测')
plt.show()

print("\n=== 趋势分解 ===")
from prophet.plot import plot_plotly
import plotly.graph_objs as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='实际值'))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='趋势'))
fig.show()

print("\n=== 交叉验证 ===")
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(m, initial='365 days', period='30 days', horizon='90 days')
print(f"交叉验证结果形状: {df_cv.shape}")

df_performance = performance_metrics(df_cv)
print(f"平均RMSE: {df_performance['rmse'].mean():.4f}")
print(f"平均MAE: {df_performance['mae'].mean():.4f}")

print("\n=== 参数调优 ===")
from prophet.diagnostics import grid_search

param_grid = {
    'changepoint_prior_scale': [0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.1, 1.0, 10.0]
}

results = grid_search(m, param_grid, initial='365 days', period='30 days', horizon='90 days')
print(f"最佳参数: {results.iloc[0].to_dict()}")

print("\n=== 预测区间调整 ===")
m_interval = Prophet(interval_width=0.95, uncertainty_samples=1000)
m_interval.fit(df)

forecast_interval = m_interval.predict(future)
print(f"预测区间宽度: 95%")