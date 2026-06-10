# PySpark SparkSQL与MLlib学习
# 主要内容：SparkSQL查询、DataFrame操作、MLlib机器学习

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sum, count
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("=== 创建SparkSession ===")
spark = SparkSession.builder \
    .appName("SparkSQL and MLlib Demo") \
    .getOrCreate()

print("\n=== 创建DataFrame ===")
data = [
    ("Alice", 25, "New York", 50000),
    ("Bob", 30, "London", 65000),
    ("Charlie", 35, "Paris", 75000),
    ("David", 28, "New York", 55000),
    ("Eve", 40, "London", 80000)
]
df = spark.createDataFrame(data, ["name", "age", "city", "salary"])
df.show()

print("\n=== SparkSQL查询 ===")
df.createOrReplaceTempView("people")

result = spark.sql("SELECT name, age, salary FROM people WHERE salary > 60000")
result.show()

print("\n=== 聚合查询 ===")
result = spark.sql("SELECT city, COUNT(*) as count, AVG(salary) as avg_salary FROM people GROUP BY city")
result.show()

print("\n=== DataFrame操作 ===")
filtered = df.filter(df.salary > 60000)
filtered.show()

grouped = df.groupBy("city").agg(
    count("name").alias("count"),
    avg("salary").alias("avg_salary")
)
grouped.show()

print("\n=== 机器学习示例 ===")
data = [
    (1.0, 2.0, 0),
    (2.0, 3.0, 0),
    (3.0, 4.0, 1),
    (4.0, 5.0, 1),
    (5.0, 6.0, 1),
    (0.5, 1.0, 0),
    (6.0, 7.0, 1)
]
df_ml = spark.createDataFrame(data, ["feature1", "feature2", "label"])

assembler = VectorAssembler(
    inputCols=["feature1", "feature2"],
    outputCol="features"
)
df_ml = assembler.transform(df_ml)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
scaler_model = scaler.fit(df_ml)
df_ml = scaler_model.transform(df_ml)

df_ml.show()

print("\n=== 训练逻辑回归模型 ===")
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")
lr_model = lr.fit(df_ml)
print("逻辑回归模型训练完成")

print("\n=== 模型评估 ===")
predictions = lr_model.transform(df_ml)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"模型准确率: {accuracy:.4f}")

print("\n=== 查看预测结果 ===")
predictions.select("features", "label", "prediction").show()

print("\n=== 停止SparkSession ===")
spark.stop()
print("SparkSession已停止")