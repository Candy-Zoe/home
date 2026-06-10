# scikit-learn模型融合学习
# 主要内容：Bagging、Boosting、Stacking、Voting、Blending

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n=== Bagging ===")
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(f"Bagging准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== AdaBoost ===")
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print(f"AdaBoost准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Gradient Boosting ===")
gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
print(f"Gradient Boosting准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Random Forest ===")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Voting Classifier ===")
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, random_state=42)
clf3 = SVC(probability=True, random_state=42)

voting = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
    voting='soft'
)
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(f"Voting准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Stacking Classifier ===")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(random_state=42)
)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print(f"Stacking准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== 特征重要性 ===")
import matplotlib.pyplot as plt

feature_importance = rf.feature_importances_
features = iris.feature_names

plt.barh(features, feature_importance)
plt.title('Random Forest特征重要性')
plt.show()