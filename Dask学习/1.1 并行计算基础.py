# Dask并行计算基础学习
# 主要内容：Dask数组、DataFrame、延迟计算

# 导入必要的库
import dask.array as da
import dask.dataframe as dd
import dask.delayed
import numpy as np
import pandas as pd

# Dask数组
print("=== Dask数组 ===")

# 创建Dask数组（从NumPy数组）
np_array = np.random.rand(1000, 1000)
dask_array = da.from_array(np_array, chunks=(100, 100))

print(f"Dask数组形状: {dask_array.shape}")
print(f"Dask数组块大小: {dask_array.chunks}")
print(f"Dask数组类型: {type(dask_array)}")

# 执行计算
result = dask_array.sum()
print(f"延迟计算结果对象: {result}")
print(f"实际计算结果: {result.compute()}")

# 创建随机Dask数组
print("\n创建随机Dask数组:")
random_array = da.random.random((1000, 1000), chunks=(250, 250))
print(f"随机数组形状: {random_array.shape}")
print(f"随机数组块大小: {random_array.chunks}")

# Dask数组操作
print("\nDask数组操作:")

# 求和
sum_result = random_array.sum().compute()
print(f"求和结果: {sum_result:.2f}")

# 均值
mean_result = random_array.mean().compute()
print(f"均值: {mean_result:.6f}")

# 标准差
std_result = random_array.std().compute()
print(f"标准差: {std_result:.6f}")

# 矩阵操作
print("\n矩阵操作:")

# 创建两个矩阵
A = da.random.random((100, 100), chunks=(50, 50))
B = da.random.random((100, 100), chunks=(50, 50))

# 矩阵乘法
C = A @ B
print(f"矩阵乘法结果形状: {C.shape}")
print(f"矩阵乘法结果块大小: {C.chunks}")

# 计算结果
C_result = C.compute()
print(f"矩阵乘法结果类型: {type(C_result)}")

# Dask DataFrame
print("\n=== Dask DataFrame ===")

# 创建示例数据
n_rows = 10000
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'] * (n_rows // 4),
    'age': np.random.randint(20, 60, n_rows),
    'city': ['New York', 'London', 'Paris', 'Tokyo'] * (n_rows // 4),
    'salary': np.random.randint(5000, 20000, n_rows)
}
pdf = pd.DataFrame(data)

# 创建Dask DataFrame
ddf = dd.from_pandas(pdf, npartitions=4)

print(f"Dask DataFrame分区数: {ddf.npartitions}")
print(f"Dask DataFrame列: {list(ddf.columns)}")

# 查看数据
print("\n查看数据:")
print(ddf.head())

# 基本操作
print("\n基本操作:")

# 选择列
print("选择name和salary列:")
print(ddf[['name', 'salary']].head())

# 添加新列
ddf['salary_k'] = ddf['salary'] / 1000
print("\n添加salary_k列后:")
print(ddf[['salary', 'salary_k']].head())

# 过滤数据
filtered = ddf[ddf['age'] > 30]
print(f"\n年龄大于30的记录数: {filtered.shape[0].compute()}")

# 分组聚合
print("\n分组聚合:")

# 按城市分组统计人数和平均工资
grouped = ddf.groupby('city').agg({
    'age': 'mean',
    'salary': ['mean', 'sum']
})
print(grouped.compute())

# Dask延迟计算
print("\n=== Dask延迟计算 ===")

@dask.delayed
def add(a, b):
    """延迟添加函数"""
    return a + b

@dask.delayed
def multiply(a, b):
    """延迟乘法函数"""
    return a * b

# 创建延迟计算对象
x = add(1, 2)
y = multiply(x, 3)
z = add(y, 4)

print(f"延迟计算对象x: {x}")
print(f"延迟计算对象y: {y}")
print(f"延迟计算对象z: {z}")

# 执行计算
result = z.compute()
print(f"最终结果: {result}")

# 可视化任务图
print("\n可视化任务图:")
print(f"任务数量: {len(z.dask)}")

# Dask数据加载
print("\n=== Dask数据加载 ===")

# 从CSV加载
csv_path = 'test_data.csv'

# 创建测试CSV
pdf.to_csv(csv_path, index=False)

# 使用Dask加载CSV
ddf_csv = dd.read_csv(csv_path)
print(f"从CSV加载的DataFrame列: {list(ddf_csv.columns)}")
print(ddf_csv.head())

# 清理测试文件
import os
if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"已删除测试文件: {csv_path}")

# Dask配置
print("\n=== Dask配置 ===")

import dask
print(f"Dask版本: {dask.__version__}")

# 获取配置
config = dask.config.get()
print(f"Dask配置键数量: {len(config.keys())}")

# 设置配置
dask.config.set({'array.slicing.split_large_chunks': True})
print("配置已更新")

# Dask性能优化
print("\n=== Dask性能优化 ===")

# 块大小优化
large_array = da.random.random((10000, 10000), chunks=(1000, 1000))
print(f"大数组块大小: {large_array.chunks}")

# 预取优化
prefetch_array = da.random.random((1000, 1000), chunks=(250, 250))
result = prefetch_array.sum()

# 多线程执行
with dask.config.set(scheduler='threads'):
    result_threads = result.compute()

print(f"多线程计算结果: {result_threads:.2f}")

# 进程执行
with dask.config.set(scheduler='processes'):
    result_processes = result.compute()

print(f"多进程计算结果: {result_processes:.2f}")

print("\nDask并行计算基础学习完成！")