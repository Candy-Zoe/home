# Plotly交互式可视化学习
# 主要内容：Plotly基础、图表类型、交互功能、subplots、Dash应用

# 导入必要的库
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# 设置中文显示
import plotly.io as pio
pio.templates.default = "plotly_white"

# Plotly基础
print("=== Plotly基础 ===")

# 创建简单的折线图
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 15, 13, 17, 20],
    mode='lines+markers',
    name='系列1'
))
fig.update_layout(
    title='简单折线图',
    xaxis_title='X轴',
    yaxis_title='Y轴'
)
fig.show()

# 散点图
print("\n=== 散点图 ===")

# 创建示例数据
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'size': np.random.randint(5, 20, n),
    'color': np.random.choice(['A', 'B', 'C'], n)
})

# 使用Plotly Express创建散点图
fig = px.scatter(
    df, x='x', y='y',
    color='color',
    size='size',
    title='散点图',
    labels={'x': 'X轴', 'y': 'Y轴', 'color': '类别'}
)
fig.update_layout(width=800, height=500)
fig.show()

# 柱状图
print("\n=== 柱状图 ===")

# 分组柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [23, 45, 56, 78, 34]
values2 = [45, 32, 67, 54, 89]
values3 = [12, 28, 45, 62, 38]

fig = go.Figure()
fig.add_trace(go.Bar(name='2020', x=categories, y=values1))
fig.add_trace(go.Bar(name='2021', x=categories, y=values2))
fig.add_trace(go.Bar(name='2022', x=categories, y=values3))

fig.update_layout(
    title='分组柱状图',
    xaxis_title='类别',
    yaxis_title='值',
    barmode='group'
)
fig.show()

# 堆叠柱状图
fig = go.Figure()
fig.add_trace(go.Bar(name='产品A', x=categories, y=values1))
fig.add_trace(go.Bar(name='产品B', x=categories, y=values2))
fig.add_trace(go.Bar(name='产品C', x=categories, y=values3))

fig.update_layout(
    title='堆叠柱状图',
    xaxis_title='类别',
    yaxis_title='值',
    barmode='stack'
)
fig.show()

# 饼图
print("\n=== 饼图 ===")

labels = ['苹果', '香蕉', '橙子', '葡萄', '草莓']
values = [45, 25, 15, 10, 5]

fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.3,  # 环形图
    textinfo='label+percent',
    textposition='inside'
)])
fig.update_layout(title='水果销售分布')
fig.show()

# 饼图变体 - 玫瑰图
fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=0.5,
    marker=dict(colors=px.colors.qualitative.Set2)
)])
fig.update_traces(hoverinfo='label+percent+value')
fig.update_layout(title='玫瑰图')
fig.show()

# 直方图
print("\n=== 直方图 ===")

# 创建正态分布数据
data = np.random.randn(1000)

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=data,
    nbinsx=30,
    name='直方图',
    marker_color='steelblue'
))
fig.update_layout(
    title='直方图',
    xaxis_title='值',
    yaxis_title='频数',
    bargap=0.1
)
fig.show()

# 叠加直方图
data1 = np.random.normal(0, 1, 500)
data2 = np.random.normal(2, 1, 500)

fig = go.Figure()
fig.add_trace(go.Histogram(x=data1, name='分布1', opacity=0.7))
fig.add_trace(go.Histogram(x=data2, name='分布2', opacity=0.7))

fig.update_layout(
    title='叠加直方图',
    xaxis_title='值',
    yaxis_title='频数',
    barmode='overlay',
    bargap=0.05
)
fig.show()

# 箱线图
print("\n=== 箱线图 ===")

# 创建多组数据
data_groups = [np.random.randn(100) + i for i in range(4)]
labels = ['组A', '组B', '组C', '组D']

fig = go.Figure()
for i, (data, label) in enumerate(zip(data_groups, labels)):
    fig.add_trace(go.Box(
        y=data,
        name=label,
        boxpoints='all',  # 显示所有点
        jitter=0.3,
        pointpos=-1.8
    ))

fig.update_layout(title='箱线图')
fig.show()

# 热力图
print("\n=== 热力图 ===")

# 创建示例数据
n = 20
matrix = np.random.rand(n, n)
x_labels = [f'X{i}' for i in range(n)]
y_labels = [f'Y{i}' for i in range(n)]

fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=x_labels,
    y=y_labels,
    colorscale='Viridis',
    hoverongaps=False
))
fig.update_layout(title='热力图')
fig.show()

# 3D散点图
print("\n=== 3D图表 ===")

# 创建3D散点数据
n = 200
theta = np.linspace(0, 2*np.pi, n)
phi = np.linspace(0, np.pi, n)
x = 10 * np.outer(np.cos(theta), np.sin(phi)).flatten()
y = 10 * np.outer(np.sin(theta), np.sin(phi)).flatten()
z = 10 * np.outer(np.ones(n), np.cos(phi)).flatten()

fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=2,
        color=z,
        colorscale='Viridis',
        opacity=0.8
    )
)])
fig.update_layout(title='3D散点图')
fig.show()

# 3D曲面图
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=[go.Surface(
    x=X, y=Y, z=Z,
    colorscale='RdBu',
    colorbar=dict(title='值')
)])
fig.update_layout(title='3D曲面图')
fig.show()

# 极坐标图
print("\n=== 极坐标图 ===")

# 极坐标散点图
r = np.random.rand(100) * 2
theta = np.random.rand(100) * 2 * np.pi

fig = go.Figure(data=go.Scatterpolar(
    r=r,
    theta=theta * 180 / np.pi,  # 转换为度
    mode='markers',
    marker=dict(
        size=8,
        color=r,
        colorscale='Plasma'
    )
))
fig.update_layout(
    title='极坐标散点图',
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 2])
    )
)
fig.show()

# 雷达图
categories = ['速度', '力量', '耐力', '技术', '经验', '防守']
values1 = [85, 90, 75, 80, 95, 70]
values2 = [75, 85, 90, 85, 70, 80]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=values1,
    theta=categories,
    fill='toself',
    name='球员A'
))
fig.add_trace(go.Scatterpolar(
    r=values2,
    theta=categories,
    fill='toself',
    name='球员B'
))
fig.update_layout(
    title='雷达图',
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 100])
    ),
    showlegend=True
)
fig.show()

# 子图布局
print("\n=== 子图布局 ===")

# 创建子图
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['折线图', '散点图', '柱状图', '饼图'],
    specs=[[{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "pie"}]]
)

# 添加折线图
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines+markers'),
    row=1, col=1
)

# 添加散点图
fig.add_trace(
    go.Scatter(x=[1, 2, 3], y=[6, 5, 4], mode='markers'),
    row=1, col=2
)

# 添加柱状图
fig.add_trace(
    go.Bar(x=['A', 'B', 'C'], y=[3, 5, 2]),
    row=2, col=1
)

# 添加饼图
fig.add_trace(
    go.Pie(labels=['X', 'Y', 'Z'], values=[30, 50, 20]),
    row=2, col=2
)

fig.update_layout(height=700, title_text='多子图布局')
fig.show()

# 时间序列图
print("\n=== 时间序列图 ===")

# 创建时间序列数据
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates, y=values,
    mode='lines',
    name='值',
    line=dict(color='steelblue', width=2)
))

# 添加移动平均
ma = pd.Series(values).rolling(window=7).mean()
fig.add_trace(go.Scatter(
    x=dates, y=ma,
    mode='lines',
    name='7日均线',
    line=dict(color='red', width=2, dash='dash')
))

fig.update_layout(
    title='时间序列',
    xaxis_title='日期',
    yaxis_title='值',
    hovermode='x unified'
)
fig.show()

# 股票K线图
print("\n=== 股票K线图 ===")

# 创建示例股票数据
import pandas as pd
np.random.seed(42)
n_days = 50
stock_dates = pd.date_range('2023-01-01', periods=n_days, freq='D')

open_prices = 100 + np.cumsum(np.random.randn(n_days) * 2)
high_prices = open_prices + np.abs(np.random.randn(n_days) * 3)
low_prices = open_prices - np.abs(np.random.randn(n_days) * 3)
close_prices = open_prices + np.random.randn(n_days) * 2

fig = go.Figure(data=[go.Candlestick(
    x=stock_dates,
    open=open_prices,
    high=high_prices,
    low=low_prices,
    close=close_prices,
    name='K线'
)])

fig.update_layout(
    title='股票K线图',
    yaxis_title='价格',
    xaxis_rangeslider_visible=False
)
fig.show()

# 旭日图
print("\n=== 旭日图 ===")

import plotly.graph_objects as go

fig = go.Figure(go.Sunburst(
    labels=['全球', '亚洲', '欧洲', '北美', '中国', '日本', '印度', '德国', '法国', '英国', '美国', '加拿大'],
    parents=['', '全球', '全球', '全球', '亚洲', '亚洲', '亚洲', '欧洲', '欧洲', '欧洲', '北美', '北美'],
    values=[100, 40, 30, 30, 15, 10, 15, 10, 10, 10, 20, 10],
))
fig.update_layout(title='旭日图')
fig.show()

# 地图可视化
print("\n=== 地图可视化 ===")

# 创建示例地理数据
map_data = pd.DataFrame({
    '城市': ['北京', '上海', '广州', '深圳', '杭州'],
    'lat': [39.9042, 31.2304, 23.1291, 22.5431, 30.2741],
    'lon': [116.4074, 121.4737, 113.2644, 114.0579, 120.1551],
    '人口': [2154, 2428, 1530, 1253, 1036]
})

fig = px.scatter_geo(
    map_data,
    lat='lat',
    lon='lon',
    size='人口',
    color='城市',
    hover_name='城市',
    title='中国主要城市分布',
    projection='natural earth'
)
fig.show()

# 交互功能
print("\n=== 交互功能 ===")

# 创建带有注释的图表
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 15, 13, 17, 20],
    mode='lines+markers',
    name='数据'
))

# 添加注释
fig.add_annotation(
    x=3,
    y=13,
    text='最低点',
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor='red',
    ax=-30,
    ay=-30
)

# 添加形状
fig.add_shape(
    type='rect',
    x0=2, y0=10,
    x1=4, y1=18,
    fillcolor='lightblue',
    opacity=0.3,
    line=dict(color='blue', width=2)
)

fig.update_layout(title='带注释的图表')
fig.show()

# 自定义颜色和样式
print("\n=== 自定义颜色和样式 ===")

# 使用自定义颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

fig = go.Figure()
for i, color in enumerate(colors):
    fig.add_trace(go.Bar(
        x=['A', 'B', 'C', 'D'],
        y=[10+i*2, 15+i*2, 12+i*2, 18+i*2],
        name=f'系列{i+1}',
        marker_color=color
    ))

fig.update_layout(title='自定义颜色柱状图')
fig.show()

# 导出图表
print("\n=== 导出图表 ===")

# 创建图表
fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))

# 导出为HTML
fig.write_html('chart.html')
print("图表已导出为 chart.html")

# 导出为PNG
fig.write_image('chart.png', width=800, height=600, scale=2)
print("图表已导出为 chart.png")

# 导出为SVG
fig.write_image('chart.svg')
print("图表已导出为 chart.svg")

# 清理
import os
for f in ['chart.html', 'chart.png', 'chart.svg']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")

# 动画图表
print("\n=== 动画图表 ===")

# 创建动画数据
np.random.seed(42)
n_frames = 10
data_frames = []

for i in range(n_frames):
    df = pd.DataFrame({
        'x': range(20),
        'y': np.random.randn(20).cumsum() + i * 5,
        'frame': i
    })
    data_frames.append(df)

all_data = pd.concat(data_frames)

fig = px.line(
    all_data,
    x='x',
    y='y',
    animation_frame='frame',
    range_y=[-10, 50],
    title='动画折线图'
)

fig.update_layout(
    xaxis_title='X',
    yaxis_title='Y',
    updatemenus=[dict(type='buttons', showactive=False,
                      y=0, x=0.1, xanchor='right',
                      buttons=[dict(label='播放',
                                   method='animate',
                                   args=[None, dict(frame=dict(duration=500, redraw=True),
                                                  fromcurrent=True)])])]
)
fig.show()

# 动态添加数据
print("\n=== 动态更新图表 ===")

from plotly.subplots import make_subplots
import time

# 创建实时更新图表
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

# 添加初始数据
x_data = list(range(10))
y1_data = [0]
y2_data = [0]

fig.add_trace(go.Scatter(x=x_data, y=y1_data, name='信号1', mode='lines+markers'), row=1, col=1)
fig.add_trace(go.Scatter(x=x_data, y=y2_data, name='信号2', mode='lines+markers'), row=2, col=1)

fig.update_layout(height=500, title='实时数据监控')

# 模拟实时更新
print("实时更新图表已创建")
print("(在Jupyter环境中可以使用 %matplotlib notebook 进行交互式更新)")

# 总结
print("\n=== Plotly交互式可视化学习总结 ===")
print("1. Plotly基础和图形对象")
print("2. 多种图表类型（散点、柱状、饼图、热力图等）")
print("3. 3D可视化")
print("4. 极坐标和雷达图")
print("5. 子图布局")
print("6. 时间序列和金融图表")
print("7. 地图可视化")
print("8. 旭日图和特殊图表")
print("9. 交互功能（注释、形状、悬停）")
print("10. 动画图表")
print("11. 图表导出")

print("\nPlotly交互式可视化学习完成！")
