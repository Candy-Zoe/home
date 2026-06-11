# scikit-learn模型管道和Pipeline学习
# 主要内容：Pipeline构建、ColumnTransformer、交叉验证流程

# 导入必要的库
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# 创建示例数据集
print("=== 创建示例数据集 ===")

# 创建一个包含数值和类别特征的数据集
data = {
    '年龄': [25, 30, 35, 40, 45, None, 50, 55, 60, 65],
    '工资': [5000, 8000, 12000, 15000, 10000, 7000, 9000, 11000, 13000, 14000],
    '城市': ['北京', '上海', '北京', '深圳', '上海', '北京', '深圳', '上海', '北京', '深圳'],
    '学历': ['本科', '硕士', '本科', '博士', '本科', '硕士', '本科', '本科', '博士', '硕士'],
    '购买': [0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("原始数据:")
print(df)

# 分离特征和目标变量
X = df.drop('购买', axis=1)
y = df['购买']

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 定义数值和类别特征
numeric_features = ['年龄', '工资']
categorical_features = ['城市', '学历']

print(f"\n数值特征: {numeric_features}")
print(f"类别特征: {categorical_features}")

# Pipeline基础
print("\n=== Pipeline基础 ===")

# 创建数值预处理器
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 缺失值填充
    ('scaler', StandardScaler())  # 标准化
])

# 创建类别预处理器
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 缺失值填充
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 独热编码
])

# 创建列转换器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 创建完整的Pipeline
pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=200))
])

print("逻辑回归Pipeline:")
print(pipeline_lr)

# 使用make_pipeline简化Pipeline创建
print("\n使用make_pipeline:")
pipeline_dt = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    DecisionTreeClassifier(random_state=42)
)
print(pipeline_dt)

# 使用Pipeline进行训练和预测
print("\n=== 训练和预测 ===")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练Pipeline
pipeline_lr.fit(X_train, y_train)
print("Pipeline训练完成")

# 预测
y_pred = pipeline_lr.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# Pipeline获取中间步骤
print("\n=== 获取Pipeline中间步骤 ===")

# 获取预处理后的特征
X_train_transformed = pipeline_lr.named_steps['preprocessor'].transform(X_train)
print(f"预处理后特征形状: {X_train_transformed.shape}")

# 获取特征名称
feature_names = pipeline_lr.named_steps['preprocessor'].get_feature_names_out()
print(f"\n特征名称:")
for name in feature_names:
    print(f"  {name}")

# Pipeline交叉验证
print("\n=== Pipeline交叉验证 ===")

# 对整个Pipeline进行交叉验证
cv_scores = cross_val_score(pipeline_lr, X, y, cv=3, scoring='accuracy')
print(f"3折交叉验证准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 不同模型的Pipeline对比
print("\n=== 不同模型的Pipeline对比 ===")

# 定义多个Pipeline
pipelines = {
    '逻辑回归': make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        LogisticRegression(random_state=42, max_iter=200)
    ),
    '决策树': make_pipeline(
        SimpleImputer(strategy='median'),
        DecisionTreeClassifier(random_state=42, max_depth=3)
    ),
    '随机森林': make_pipeline(
        SimpleImputer(strategy='median'),
        RandomForestClassifier(random_state=42, n_estimators=100)
    )
}

# 评估每个Pipeline
results = []
for name, pipeline in pipelines.items():
    # 交叉验证
    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy')
    results.append({
        '模型': name,
        '平均准确率': cv_scores.mean(),
        '标准差': cv_scores.std()
    })
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 结果汇总
results_df = pd.DataFrame(results)
print("\n模型对比结果:")
print(results_df)

# 自定义转换器
print("\n=== 自定义转换器 ===")

from sklearn.base import BaseEstimator, TransformerMixin

class AgeBinningTransformer(BaseEstimator, TransformerMixin):
    """年龄分箱转换器"""

    def __init__(self, n_bins=3):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        # 计算分箱边界
        self.bins_ = np.percentile(X['年龄'].dropna(), 
                                    np.linspace(0, 100, self.n_bins + 1))
        return self

    def transform(self, X):
        X = X.copy()
        X['年龄_分箱'] = pd.cut(X['年龄'], bins=self.bins_, labels=False)
        return X

# 创建包含自定义转换器的Pipeline
pipeline_custom = make_pipeline(
    AgeBinningTransformer(n_bins=4),
    SimpleImputer(strategy='median'),
    StandardScaler(),
    LogisticRegression(random_state=42, max_iter=200)
)

# 训练自定义Pipeline
X_train_transformed = X_train.copy()
pipeline_custom.fit(X_train_transformed, y_train)
print("自定义Pipeline训练完成")

# Pipeline保存和加载
print("\n=== Pipeline保存和加载 ===")

import joblib

# 保存Pipeline
joblib.dump(pipeline_lr, 'logistic_regression_pipeline.pkl')
print("Pipeline已保存")

# 加载Pipeline
loaded_pipeline = joblib.load('logistic_regression_pipeline.pkl')
print("Pipeline已加载")

# 使用加载的Pipeline进行预测
y_pred_loaded = loaded_pipeline.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"加载Pipeline的准确率: {accuracy_loaded:.4f}")

# 使用make_column_transformer
print("\n=== make_column_transformer ===")

# 使用make_column_transformer创建列转换器
preprocessor2 = make_column_transformer(
    (StandardScaler(), ['年龄', '工资']),
    (OneHotEncoder(handle_unknown='ignore'), ['城市', '学历'])
)

print("列转换器:")
print(preprocessor2)

# ColumnTransformer的高级用法
print("\n=== ColumnTransformer高级用法 ===")

# 创建带命名的预处理器
preprocessor_advanced = ColumnTransformer(
    transformers=[
        ('numeric_features', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('categorical_features', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ],
    remainder='drop'  # 其他列的处理方式
)

# 创建Pipeline
pipeline_advanced = Pipeline(steps=[
    ('preprocessor', preprocessor_advanced),
    ('classifier', LogisticRegression(random_state=42, max_iter=200))
])

# 交叉验证
cv_scores_advanced = cross_val_score(pipeline_advanced, X, y, cv=3)
print(f"高级Pipeline交叉验证: {cv_scores_advanced.mean():.4f}")

# 完整的机器学习流程示例
print("\n=== 完整的机器学习流程 ===")

# 1. 准备数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. 创建Pipeline
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_advanced),
    ('classifier', RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        random_state=42
    ))
])

# 3. 训练
final_pipeline.fit(X_train, y_train)

# 4. 预测
y_pred_final = final_pipeline.predict(X_test)

# 5. 评估
print("\n最终模型评估:")
print(f"准确率: {accuracy_score(y_test, y_pred_final):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_final))

# 获取特征重要性
feature_importance = final_pipeline.named_steps['classifier'].feature_importances_
print(f"\n特征重要性:")
for name, importance in sorted(zip(feature_names, feature_importance), 
                                 key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.4f}")

# Pipeline的网格搜索
print("\n=== Pipeline的网格搜索 ===")

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'preprocessor__numeric_features__scaler': [StandardScaler(), MinMaxScaler()],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 7]
}

# 创建GridSearchCV
grid_search = GridSearchCV(
    final_pipeline,
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

# 执行网格搜索
grid_search.fit(X, y)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

print("\n学习完成！")