# scikit-learn分类算法学习
# 主要内容：逻辑回归、K近邻、决策树、随机森林

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 逻辑回归 ===")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== K近邻算法 ===")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== 决策树 ===")
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== 随机森林 ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"特征重要性: {rf.feature_importances_}")