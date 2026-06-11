# scikit-learn数据预处理学习
# 主要内容：数据加载、缺失值处理、特征缩放、类别编码

# 导入必要的库
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, 
    OneHotEncoder, LabelEncoder,
    PolynomialFeatures
)
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# 加载数据集
print("=== 加载数据集 ===")

# 加载鸢尾花数据集（分类问题）
iris = load_iris()
X = iris.data
y = iris.target
print(f"鸢尾花数据集形状: X={X.shape}, y={y.shape}")

# 创建包含缺失值的示例数据
print("\n=== 缺失值处理 ===")

# 创建一个包含缺失值的DataFrame
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [6, np.nan, 8, 9, 10],
    'C': [11, 12, 13, np.nan, 15]
}
df = pd.DataFrame(data)
print(f"原始数据（含缺失值）:\n{df}")

# 使用均值填充缺失值
imputer = SimpleImputer(strategy='mean')
df_filled = imputer.fit_transform(df)
print(f"\n填充后的DataFrame:\n{df_filled}")

# 特征缩放
print("\n=== 特征缩放 ===")

# 创建示例数据
X = np.array([[100, 1], [200, 2], [300, 3]])
print(f"原始数据:\n{X}")

# 标准化（均值为0，标准差为1）
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
print(f"\n标准化后:\n{X_standardized}")
print(f"均值: {X_standardized.mean(axis=0)}")
print(f"标准差: {X_standardized.std(axis=0)}")

# 归一化（缩放到[0,1]区间）
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
print(f"\n归一化后:\n{X_normalized}")

# 类别编码
print("\n=== 类别编码 ===")

# 创建类别数据
data = np.array(['cat', 'dog', 'cat', 'bird', 'dog'])
print(f"原始类别数据: {data}")

# LabelEncoder: 将类别转换为整数
label_encoder = LabelEncoder()
data_encoded = label_encoder.fit_transform(data)
print(f"LabelEncoder编码: {data_encoded}")
print(f"类别标签: {label_encoder.classes_}")

# OneHotEncoder: 将类别转换为独热编码
data_2d = data.reshape(-1, 1)
onehot_encoder = OneHotEncoder(sparse=False)
data_onehot = onehot_encoder.fit_transform(data_2d)
print(f"\nOneHotEncoder编码:\n{data_onehot}")
print(f"类别特征名: {onehot_encoder.get_feature_names_out()}")

# 多项式特征
print("\n=== 多项式特征 ===")

# 创建简单数据
X = np.array([[1, 2], [3, 4], [5, 6]])
print(f"原始特征:\n{X}")

# 生成多项式特征（degree=2）
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"\n多项式特征（degree=2）:\n{X_poly}")
print(f"特征名称: {poly.get_feature_names_out()}")

# 数据划分
print("\n=== 数据划分 ===")

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"训练集: X={X_train.shape}, y={y_train.shape}")
print(f"测试集: X={X_test.shape}, y={y_test.shape}")

# 完整的预处理管道示例
print("\n=== 完整预处理示例 ===")

# 创建包含各种问题的数据
data = pd.DataFrame({
    'numeric': [10, 20, np.nan, 40, 50],
    'category': ['A', 'B', 'A', 'C', 'B'],
    'target': [0, 1, 0, 1, 0]
})

print(f"原始数据:\n{data}")

# 分离特征和目标
X = data.drop('target', axis=1)
y = data['target']

# 处理缺失值
imputer = SimpleImputer(strategy='median')
X['numeric'] = imputer.fit_transform(X[['numeric']])

# 编码类别特征
encoder = OneHotEncoder(sparse=False, drop='first')
encoded = encoder.fit_transform(X[['category']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# 合并处理后的数据
X_processed = pd.concat([X[['numeric']], encoded_df], axis=1)
print(f"\n处理后的数据:\n{X_processed}")