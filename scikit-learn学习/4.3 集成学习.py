# scikit-learn集成学习学习
# 主要内容：Bagging、Boosting、Stacking、Voting

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== Bagging ===")
bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10, random_state=42)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)
print(f"Bagging准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== AdaBoost ===")
adaboost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)
y_pred = adaboost.predict(X_test)
print(f"AdaBoost准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Voting ===")
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('lr', LogisticRegression(max_iter=200)),
    ('svm', SVC())
]
voting = VotingClassifier(estimators=estimators, voting='hard')
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(f"Voting准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== Stacking ===")
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('svm', SVC())
]
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=200))
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print(f"Stacking准确率: {accuracy_score(y_test, y_pred):.4f}")

print("\n=== 特征重要性 ===")
import matplotlib.pyplot as plt
feature_importance = adaboost.feature_importances_
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('AdaBoost特征重要性')
plt.show()