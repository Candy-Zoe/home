# Plotly交互式可视化学习
# 主要内容：交互式图表、仪表盘、动画、子图

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

print("=== 交互式散点图 ===")
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()

print("\n=== 交互式折线图 ===")
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='sin(x)'))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='cos(x)'))
fig.update_layout(title='三角函数', xaxis_title='x', yaxis_title='y')
fig.show()

print("\n=== 交互式柱状图 ===")
fig = px.bar(df, x="species", y="sepal_width", color="species", barmode="group")
fig.show()

print("\n=== 3D可视化 ===")
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species', size='petal_length')
fig.show()

print("\n=== 仪表盘 ===")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=75,
    title={'text': "完成度"},
    gauge={'axis': {'range': [None, 100]},
           'bar': {'color': "darkblue"},
           'steps': [
               {'range': [0, 50], 'color': "red"},
               {'range': [50, 80], 'color': "yellow"},
               {'range': [80, 100], 'color': "green"}]
           }))
fig.show()

print("\n=== 子图 ===")
fig = make_subplots(rows=2, cols=2, subplot_titles=("散点图", "折线图", "柱状图", "直方图"))

fig.add_trace(go.Scatter(x=df['sepal_width'], y=df['sepal_length'], mode='markers'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=np.sin(x), mode='lines'), row=1, col=2)
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[3, 5, 2]), row=2, col=1)
fig.add_trace(go.Histogram(x=df['sepal_length']), row=2, col=2)

fig.update_layout(height=600, width=800, title_text="子图示例")
fig.show()

print("\n=== 动画 ===")
df['size'] = df['petal_length'] * 10
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='size', animation_frame="species",
                 range_x=[2, 5], range_y=[4, 8])
fig.show()

print("\n=== 热力图 ===")
corr_matrix = df.corr()
fig = px.imshow(corr_matrix, text_auto=True)
fig.show()