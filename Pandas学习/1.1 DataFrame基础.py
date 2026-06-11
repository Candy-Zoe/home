# Pandas DataFrame基础学习
# 主要内容：DataFrame的创建、基本属性、查看数据

# 导入Pandas库，通常使用pd作为别名
import pandas as pd
import numpy as np

# 创建DataFrame的方法
print("=== 创建DataFrame ===")

# 方法1：使用字典创建
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 35, 40],
    '城市': ['北京', '上海', '广州', '深圳'],
    '工资': [8000, 12000, 10000, 15000]
}
df = pd.DataFrame(data)
print(f"DataFrame内容:\n{df}")

# 方法2：使用二维数组创建，指定列名
data_array = np.array([
    ['Alice', 23, 'New York', 6000],
    ['Bob', 28, 'London', 7500],
    ['Charlie', 32, 'Paris', 8000]
])
df2 = pd.DataFrame(data_array, columns=['姓名', '年龄', '城市', '工资'])
print(f"\n使用数组创建的DataFrame:\n{df2}")

# DataFrame的基本属性
print("\n=== DataFrame属性 ===")

# shape: 返回DataFrame的形状（行数, 列数）
print(f"DataFrame形状: {df.shape}")

# columns: 返回列名列表
print(f"列名: {df.columns.tolist()}")

# index: 返回行索引
print(f"行索引: {df.index.tolist()}")

# dtypes: 返回各列的数据类型
print(f"数据类型:\n{df.dtypes}")

# 查看DataFrame的内容
print("\n=== 查看数据 ===")

# head(): 查看前几行（默认前5行）
print("前3行数据:")
print(df.head(3))

# tail(): 查看后几行（默认后5行）
print("\n后2行数据:")
print(df.tail(2))

# info(): 查看DataFrame的基本信息，包括非空值数量和内存使用
print("\nDataFrame信息:")
df.info()

# describe(): 查看数值列的统计信息
print("\n统计摘要:")
print(df.describe())

# 访问数据
print("\n=== 访问数据 ===")

# 通过列名访问整列
print(f"访问'姓名'列:\n{df['姓名']}")

# 访问多列
print(f"\n访问'姓名'和'工资'列:\n{df[['姓名', '工资']]}")

# 通过loc访问指定行和列（基于标签）
print(f"\n访问第0行所有列:\n{df.loc[0]}")

# 通过iloc访问指定位置的行和列（基于索引）
print(f"\n访问前2行的前2列:\n{df.iloc[:2, :2]}")

# 统计计算
print("\n=== 统计计算 ===")

# 计算某列的均值
print(f"平均年龄: {df['年龄'].mean()}")

# 计算某列的总和
print(f"总工资: {df['工资'].sum()}")

# 计算某列的标准差
print(f"年龄标准差: {df['年龄'].std()}")

# 计算相关系数矩阵
print(f"\n相关系数矩阵:\n{df.corr()}")