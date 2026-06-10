# Plotly高级图表学习
# 主要内容：热力图、3D图、子图、动画

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=== 热力图 ===")
flights = px.data.flights()
fig = px.density_heatmap(flights, x='year', y='month', z='passengers')
fig.show()

print("\n=== 3D散点图 ===")
iris = px.data.iris()
fig = px.scatter_3d(iris, x='sepal_length', y='sepal_width', z='petal_length', color='species')
fig.show()

print("\n=== 子图 ===")
tips = px.data.tips()
fig = make_subplots(rows=1, cols=2, subplot_titles=('散点图', '直方图'))

fig.add_trace(go.Scatter(x=tips['total_bill'], y=tips['tip'], mode='markers'), row=1, col=1)
fig.add_trace(go.Histogram(x=tips['total_bill']), row=1, col=2)

fig.update_layout(title_text='消费数据可视化')
fig.show()

print("\n=== 动画图表 ===")
gapminder = px.data.gapminder()
fig = px.scatter(gapminder, x='gdpPercap', y='lifeExp', animation_frame='year', 
                 size='pop', color='continent', hover_name='country')
fig.show()

print("\n=== 箱线图 ===")
fig = px.box(tips, x='day', y='total_bill', color='sex')
fig.show()