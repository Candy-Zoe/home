# Pandas数据选择与过滤学习
# 主要内容：列选择、行选择、条件过滤、混合选择

# 导入Pandas库
import pandas as pd
import numpy as np

# 创建示例DataFrame
print("=== 创建示例数据 ===")

data = {
    '姓名': ['张三', '李四', '王五', '赵六', '陈七'],
    '年龄': [25, 30, 35, 28, 42],
    '城市': ['北京', '上海', '北京', '深圳', '上海'],
    '工资': [8000, 12000, 10000, 15000, 9000],
    '部门': ['技术', '销售', '技术', '人事', '销售']
}
df = pd.DataFrame(data)
print(df)

# 列选择
print("\n=== 列选择 ===")

# 选择单列（返回Series）
print("选择'姓名'列:")
print(df['姓名'])
print(f"类型: {type(df['姓名'])}")

# 选择多列（返回DataFrame）
print("\n选择多列 ['姓名', '工资']:")
print(df[['姓名', '工资']])
print(f"类型: {type(df[['姓名', '工资']])}")

# 使用点号访问列（不推荐，可能与DataFrame方法冲突）
print("\n使用点号访问 df.姓名:")
print(df.姓名)

# 行选择
print("\n=== 行选择 ===")

# 使用iloc按位置选择
print("使用iloc[0]选择第一行:")
print(df.iloc[0])

print("\n使用iloc[0:3]选择前3行:")
print(df.iloc[0:3])

print("\n使用iloc[[0, 2, 4]]选择第1,3,5行:")
print(df.iloc[[0, 2, 4]])

# 使用loc按标签选择
print("\n使用loc[0]选择第一行:")
print(df.loc[0])

print("\n使用loc[0:2]选择前3行（包含结束标签）:")
print(df.loc[0:2])

# 选择特定行和列
print("\n=== 混合选择 ===")

print("选择第0行和第2行，'姓名'和'工资'列:")
print(df.loc[[0, 2], ['姓名', '工资']])

print("\n选择前2行，前3列:")
print(df.iloc[:2, :3])

# 条件过滤
print("\n=== 条件过滤 ===")

# 单条件过滤
print("年龄大于30的员工:")
print(df[df['年龄'] > 30])

# 多条件过滤
print("\n年龄大于30且工资大于10000:")
print(df[(df['年龄'] > 30) & (df['工资'] > 10000)])

# 使用|表示或
print("\n城市为'北京'或'上海'的员工:")
print(df[(df['城市'] == '北京') | (df['城市'] == '上海')])

# 使用==进行字符串匹配
print("\n姓名为'张三'的员工:")
print(df[df['姓名'] == '张三'])

# 使用str.contains进行模糊匹配
print("\n姓名包含'三'的员工:")
print(df[df['姓名'].str.contains('三')])

# 使用isin进行列表匹配
print("\n城市在['北京', '上海']中的员工:")
print(df[df['城市'].isin(['北京', '上海'])])

# 使用query方法（更直观的语法）
print("\n使用query方法 - 年龄>30:")
print(df.query('年龄 > 30'))

print("\n使用query方法 - 多个条件:")
print(df.query('年龄 > 30 and 工资 > 10000'))

# 高级过滤
print("\n=== 高级过滤 ===")

# 使用between筛选范围
print("年龄在25-35之间的员工:")
print(df[df['年龄'].between(25, 35)])

# 使用isnull/isnotnull处理空值
df_with_na = df.copy()
df_with_na.loc[2, '工资'] = np.nan
print("\n包含空值的DataFrame:")
print(df_with_na)

print("\n选择工资不为空的行:")
print(df_with_na[df_with_na['工资'].notna()])

# 使用nlargest/nsmallest
print("\n工资最高的3名员工:")
print(df.nlargest(3, '工资'))

print("\n年龄最小的2名员工:")
print(df.nsmallest(2, '年龄'))

# 使用filter方法
print("\n=== 使用filter方法 ===")

print("列名包含'名'的列:")
print(df.filter(like='名'))

print("\n列名以'年'开头的列:")
print(df.filter(like='年'))

# 使用正则表达式
print("\n列名匹配正则表达式 '.*龄|.*名':")
print(df.filter(regex='.*龄|.*名'))

# 使用apply进行复杂过滤
print("\n=== 使用apply进行复杂过滤 ===")

def high_salary_and_young(row):
    """高工资且年轻"""
    return row['工资'] > 10000 and row['年龄'] < 35

print("高工资且年轻的员工（使用apply）:")
print(df[df.apply(high_salary_and_young, axis=1)])

# 链式过滤
print("\n=== 链式过滤 ===")

result = (df
          .query('城市 == "北京" or 城市 == "上海"')
          .query('工资 > 8000')
          [['姓名', '城市', '工资']])

print("链式过滤结果（城市为北京或上海，且工资>8000）:")
print(result)

# 设置和重置索引
print("\n=== 索引操作 ===")

# 设置索引
df_indexed = df.set_index('姓名')
print("设置姓名为主索引:")
print(df_indexed)

# 使用loc通过索引选择
print("\n选择索引为'张三'的行:")
print(df_indexed.loc['张三'])

# 重置索引
print("\n重置索引:")
print(df_indexed.reset_index())