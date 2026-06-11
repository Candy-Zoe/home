# Plotly基础图表学习
# 主要内容：Plotly Express和Graph Objects绑制交互式图表

# 导入Plotly库
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# 创建示例数据
print("=== 创建示例数据 ===")

# 创建包含中文的示例数据
data = pd.DataFrame({
    '月份': ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'],
    '月份数字': list(range(1, 13)),
    '销售额': [120, 135, 150, 180, 200, 250, 280, 260, 220, 190, 160, 140],
    '利润': [30, 35, 40, 50, 60, 80, 90, 85, 70, 55, 45, 35],
    '产品A': [50, 55, 60, 70, 80, 100, 110, 100, 90, 75, 65, 55],
    '产品B': [40, 45, 50, 60, 70, 90, 100, 95, 80, 70, 55, 45],
    '产品C': [30, 35, 40, 50, 50, 60, 70, 65, 50, 45, 40, 40],
    '城市': ['北京', '上海', '北京', '深圳', '上海', '北京', '深圳', '上海', '北京', '深圳', '上海', '北京']
})

print("示例数据:")
print(data)

# 使用Plotly Express创建图表
print("\n=== 折线图 ===")

# 创建简单折线图
fig = px.line(
    data, 
    x='月份数字', 
    y='销售额',
    title='月度销售额趋势',
    labels={'月份数字': '月份', '销售额': '销售额(万元)'}
)
fig.update_layout(template='plotly_white')
fig.show()

# 创建多条折线
fig = px.line(
    data, 
    x='月份数字', 
    y=['产品A', '产品B', '产品C'],
    title='各产品月度销售趋势',
    labels={'月份数字': '月份', 'value': '销售额', 'variable': '产品'}
)
fig.show()

# 散点图
print("\n=== 散点图 ===")

fig = px.scatter(
    data, 
    x='销售额', 
    y='利润',
    size='月份数字',
    color='城市',
    title='销售额与利润关系',
    labels={'销售额': '销售额(万元)', '利润': '利润(万元)'}
)
fig.update_traces(marker=dict(size=12))
fig.show()

# 条形图
print("\n=== 条形图 ===")

fig = px.bar(
    data, 
    x='月份', 
    y='销售额',
    color='销售额',
    title='月度销售额条形图',
    labels={'月份': '月份', '销售额': '销售额(万元)'}
)
fig.show()

# 分组条形图
fig = px.bar(
    data, 
    x='月份', 
    y=['产品A', '产品B', '产品C'],
    title='各产品月度销售对比',
    barmode='group'
)
fig.show()

# 饼图
print("\n=== 饼图 ===")

# 计算各城市总销售额
city_sales = data.groupby('城市')['销售额'].sum().reset_index()

fig = px.pie(
    city_sales, 
    values='销售额', 
    names='城市',
    title='各城市销售占比',
    hole=0.3  # 环形图
)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()

# 箱线图
print("\n=== 箱线图 ===")

fig = px.box(
    data, 
    x='城市', 
    y='销售额',
    title='各城市销售额分布',
    points='all'  # 显示所有数据点
)
fig.show()

# 热力图
print("\n=== 热力图 ===")

# 创建数据矩阵
sales_matrix = data[['产品A', '产品B', '产品C']].values

fig = go.Figure(data=go.Heatmap(
    z=sales_matrix,
    x=['产品A', '产品B', '产品C'],
    y=data['月份'],
    colorscale='Viridis',
    colorbar=dict(title='销售额')
))
fig.update_layout(title='月度产品销售热力图')
fig.show()

# 使用Graph Objects创建更复杂的图表
print("\n=== 使用Graph Objects ===")

# 创建带有误差线的图表
fig = go.Figure()

# 添加销售额曲线
fig.add_trace(go.Scatter(
    x=data['月份'],
    y=data['销售额'],
    mode='lines+markers',
    name='销售额',
    line=dict(color='blue', width=2),
    error_y=dict(
        type='data',
        array=[15, 18, 20, 22, 25, 30, 35, 32, 28, 25, 20, 18],
        visible=True,
        color='blue'
    )
))

# 添加利润曲线
fig.add_trace(go.Scatter(
    x=data['月份'],
    y=data['利润'],
    mode='lines+markers',
    name='利润',
    line=dict(color='green', width=2),
    error_y=dict(
        type='data',
        array=[5, 6, 7, 8, 9, 10, 12, 11, 9, 8, 7, 6],
        visible=True,
        color='green'
    )
))

fig.update_layout(
    title='销售额与利润趋势（含误差线）',
    xaxis_title='月份',
    yaxis_title='金额（万元）',
    template='plotly_white'
)
fig.show()

# 子图布局
print("\n=== 子图布局 ===")

from plotly.subplots import make_subplots

# 创建2x2子图
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['销售额趋势', '利润趋势', '产品对比', '城市分布'],
    specs=[[{'type': 'scatter'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'pie'}]]
)

# 添加第一个子图：销售额趋势
fig.add_trace(
    go.Scatter(x=data['月份'], y=data['销售额'], mode='lines+markers'),
    row=1, col=1
)

# 添加第二个子图：利润趋势
fig.add_trace(
    go.Bar(x=data['月份'], y=data['利润'], marker_color='green'),
    row=1, col=2
)

# 添加第三个子图：产品对比
fig.add_trace(
    go.Bar(x=data['月份'], y=data['产品A'], name='产品A'),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=data['月份'], y=data['产品B'], name='产品B'),
    row=2, col=1
)

# 添加第四个子图：城市分布饼图
fig.add_trace(
    go.Pie(values=city_sales['销售额'], labels=city_sales['城市']),
    row=2, col=2
)

fig.update_layout(height=800, width=1000, title_text='销售数据分析仪表板', showlegend=False)
fig.show()

# 3D图表
print("\n=== 3D图表 ===")

# 创建3D散点图
fig = px.scatter_3d(
    data,
    x='产品A',
    y='产品B',
    z='产品C',
    color='城市',
    size='销售额',
    title='产品销售3D分布'
)
fig.show()