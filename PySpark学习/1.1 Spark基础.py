# PySpark基础学习
# 主要内容：SparkSession创建、RDD操作、DataFrame基础

# 导入必要的库
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import col, sum, avg, count

# 创建SparkSession
print("=== 创建SparkSession ===")

# 创建SparkSession实例
spark = SparkSession.builder \
    .appName("PySpark基础学习") \
    .master("local[*]") \
    .getOrCreate()

print(f"Spark版本: {spark.version}")
print(f"Spark应用名称: {spark.sparkContext.appName}")
print(f"Spark配置: {spark.sparkContext.getConf().getAll()[:5]}")

# 创建RDD
print("\n=== 创建RDD ===")

# 方法1：从列表创建
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rdd = spark.sparkContext.parallelize(data)

print(f"RDD元素数量: {rdd.count()}")
print(f"RDD前5个元素: {rdd.take(5)}")
print(f"RDD最大值: {rdd.max()}")
print(f"RDD最小值: {rdd.min()}")
print(f"RDD求和: {rdd.sum()}")
print(f"RDD平均值: {rdd.mean()}")

# RDD转换操作
print("\n=== RDD转换操作 ===")

# map操作：对每个元素乘以2
rdd_double = rdd.map(lambda x: x * 2)
print(f"map操作后前5个元素: {rdd_double.take(5)}")

# filter操作：筛选偶数
rdd_even = rdd.filter(lambda x: x % 2 == 0)
print(f"filter操作后元素: {rdd_even.collect()}")

# flatMap操作
rdd_words = spark.sparkContext.parallelize(["Hello World", "Spark is great"])
rdd_flat = rdd_words.flatMap(lambda x: x.split())
print(f"flatMap操作后: {rdd_flat.collect()}")

# reduce操作
rdd_sum = rdd.reduce(lambda a, b: a + b)
print(f"reduce求和: {rdd_sum}")

# 创建DataFrame
print("\n=== 创建DataFrame ===")

# 方法1：从RDD创建
data_rdd = spark.sparkContext.parallelize([
    ("Alice", 25, "New York"),
    ("Bob", 30, "London"),
    ("Charlie", 35, "Paris"),
    ("David", 28, "New York")
])

df = data_rdd.toDF(["name", "age", "city"])
print("从RDD创建的DataFrame:")
df.show()

# 方法2：从列表创建
data_list = [
    Row(name="Alice", age=25, city="New York"),
    Row(name="Bob", age=30, city="London"),
    Row(name="Charlie", age=35, city="Paris")
]
df2 = spark.createDataFrame(data_list)
print("\n从列表创建的DataFrame:")
df2.show()

# 方法3：从Pandas DataFrame创建
import pandas as pd
pdf = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "city": ["New York", "London", "Paris"]
})
df3 = spark.createDataFrame(pdf)
print("\n从Pandas创建的DataFrame:")
df3.show()

# DataFrame基本操作
print("\n=== DataFrame基本操作 ===")

# 查看DataFrame结构
print("DataFrame结构:")
df.printSchema()

# 查看前几行
print("\nDataFrame前2行:")
df.head(2)

# 选择列
print("\n选择name列:")
df.select("name").show()

# 选择多列
print("\n选择name和age列:")
df.select("name", "age").show()

# 添加新列
df_with_salary = df.withColumn("salary", col("age") * 1000)
print("\n添加salary列后:")
df_with_salary.show()

# 修改列名
df_renamed = df.withColumnRenamed("city", "city_name")
print("\n修改列名后:")
df_renamed.show()

# 过滤数据
print("\n过滤年龄大于30的记录:")
df.filter(col("age") > 30).show()

# 排序
print("\n按年龄降序排序:")
df.orderBy(col("age").desc()).show()

# 分组聚合
print("\n=== 分组聚合 ===")

# 按城市分组统计人数和平均年龄
df.groupBy("city") \
    .agg(count("name").alias("人数"), avg("age").alias("平均年龄")) \
    .show()

# 按城市分组统计总年龄
df.groupBy("city") \
    .sum("age") \
    .show()

# SQL查询
print("\n=== SQL查询 ===")

# 创建临时视图
df.createOrReplaceTempView("people")

# 执行SQL查询
result = spark.sql("SELECT name, age FROM people WHERE age > 30")
print("SQL查询结果:")
result.show()

# 更复杂的SQL查询
result2 = spark.sql("""
    SELECT city, COUNT(*) as count, AVG(age) as avg_age
    FROM people
    GROUP BY city
    ORDER BY count DESC
""")
print("\nSQL分组查询结果:")
result2.show()

# 文件读写
print("\n=== 文件读写 ===")

# 保存为Parquet格式
df.write.mode("overwrite").parquet("people.parquet")
print("DataFrame已保存为Parquet格式")

# 读取Parquet文件
df_parquet = spark.read.parquet("people.parquet")
print("\n从Parquet读取的DataFrame:")
df_parquet.show()

# 保存为CSV格式
df.write.mode("overwrite").csv("people.csv", header=True)
print("\nDataFrame已保存为CSV格式")

# 读取CSV文件
df_csv = spark.read.csv("people.csv", header=True, inferSchema=True)
print("\n从CSV读取的DataFrame:")
df_csv.show()

# 清理临时文件
import os
import shutil
for f in ['people.parquet', 'people.csv']:
    if os.path.exists(f):
        if os.path.isdir(f):
            shutil.rmtree(f)
        else:
            os.remove(f)
        print(f"已删除: {f}")

# 关闭SparkSession
spark.stop()
print("\nSparkSession已关闭")

print("\nPySpark基础学习完成！")