# PySpark基础学习
# 主要内容：SparkSession创建、RDD操作、DataFrame基础

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

print("=== 创建SparkSession ===")
spark = SparkSession.builder \
    .appName("PySpark学习") \
    .getOrCreate()

print(f"Spark版本: {spark.version}")

print("\n=== 创建DataFrame ===")
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
df = spark.createDataFrame(data, ["name", "age"])
df.show()

print("\n=== DataFrame基本操作 ===")
print("Schema:")
df.printSchema()

print("\n选择列:")
df.select("name").show()

print("\n过滤数据:")
df.filter(df.age > 30).show()

print("\n聚合操作:")
df.groupBy().avg("age").show()

print("\n创建临时视图:")
df.createOrReplaceTempView("people")
result = spark.sql("SELECT * FROM people WHERE age > 30")
result.show()

print("\n=== 读取CSV文件 ===")
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])
csv_df = spark.createDataFrame(data, schema)
csv_df.write.csv("people.csv", header=True)
print("CSV文件已保存")

print("\n=== 停止SparkSession ===")
spark.stop()

print("\n=== 清理测试文件 ===")
import os
import shutil
if os.path.exists('people.csv'):
    shutil.rmtree('people.csv')
    print("已删除测试文件")