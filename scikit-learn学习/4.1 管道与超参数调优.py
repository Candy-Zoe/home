# scikit-learn管道与超参数调优学习
# 主要内容：Pipeline、GridSearchCV、交叉验证

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("=== 创建管道 ===")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"管道准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== 网格搜索 ===")
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': [1, 0.1, 0.01, 0.001],
    'svm__kernel': ['linear', 'rbf']
}

grid = GridSearchCV(pipeline, param_grid, cv=5, verbose=1)
grid.fit(X_train, y_train)

print(f"最佳参数: {grid.best_params_}")
print(f"最佳交叉验证分数: {grid.best_score_:.4f}")

y_pred_best = grid.predict(X_test)
print(f"最佳模型测试准确率: {accuracy_score(y_test, y_pred_best):.4f}")

print("\n=== 管道与其他模型 ===")
from sklearn.ensemble import RandomForestClassifier

pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

param_grid_rf = {
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30]
}

grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)
print(f"随机森林最佳参数: {grid_rf.best_params_}")
print(f"随机森林最佳分数: {grid_rf.best_score_:.4f}")