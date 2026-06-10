# PySpark DataFrame操作学习
# 主要内容：数据转换、聚合、窗口函数、连接操作

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("=== 创建SparkSession ===")
spark = SparkSession.builder \
    .appName("DataFrame操作") \
    .getOrCreate()

print("\n=== 创建示例数据 ===")
data = [
    ("Alice", "Sales", 5000),
    ("Bob", "Engineering", 7000),
    ("Charlie", "Sales", 6000),
    ("David", "Engineering", 8000),
    ("Eve", "HR", 4500)
]
df = spark.createDataFrame(data, ["name", "department", "salary"])
df.show()

print("\n=== 数据转换 ===")
df.withColumn("salary_increase", df.salary * 1.1).show()

print("\n=== 分组聚合 ===")
df.groupBy("department") \
    .agg(F.sum("salary").alias("total_salary"), 
         F.avg("salary").alias("avg_salary")) \
    .show()

print("\n=== 窗口函数 ===")
window = Window.partitionBy("department").orderBy(df.salary.desc())
df.withColumn("rank", F.rank().over(window)).show()

print("\n=== 连接操作 ===")
dept_data = [("Sales", "NYC"), ("Engineering", "SF"), ("HR", "LA")]
dept_df = spark.createDataFrame(dept_data, ["department", "location"])

df.join(dept_df, on="department", how="inner").show()

print("\n=== 用户自定义函数 ===")
upper_udf = F.udf(lambda x: x.upper())
df.withColumn("name_upper", upper_udf(df.name)).show()

print("\n=== 数据排序 ===")
df.orderBy(df.salary.desc()).show()

print("\n=== 停止SparkSession ===")
spark.stop()