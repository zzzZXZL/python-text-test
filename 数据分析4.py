```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. 读取数据
df = pd.read_csv("netflix_titles.csv")

print("原始数据大小：", df.shape)
print(df.head())

# 2. 数据清洗
df = df.drop_duplicates()
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)
df = df.dropna(subset=["release_year"])

# 3. 特征工程
df["country"] = df["country"].fillna("Unknown")
df["country"] = df["country"].apply(
    lambda x: str(x).split(",")[0].strip()
)

df["listed_in"] = df["listed_in"].fillna("Unknown")
df["genre"] = df["listed_in"].apply(
    lambda x: str(x).split(",")[0].strip()
)

df["duration_value"] = pd.to_numeric(
    df["duration"].astype("string").str.extract(r"(\d+)")[0],
    errors="coerce"
)

df["rating"] = df["rating"].fillna("Unknown")
df["type"] = df["type"].fillna("Unknown")

# 4. 选择特征和目标变量
features = [
    "type",
    "rating",
    "country",
    "genre",
    "duration_value"
]

X = df[features]
y = df["release_year"].astype(float)

categorical_features = [
    "type", "rating", "country", "genre"
]
numeric_features = ["duration_value"]

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. 数据预处理
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 7. 建立 Elastic Net 模型
model = Pipeline([
    ("preprocessor", preprocessor),
    ("elasticnet", ElasticNet(
        max_iter=30000,
        random_state=42
    ))
])

# 8. 网格搜索与交叉验证
param_grid = {
    "elasticnet__alpha": np.logspace(-3, 2, 15),
    "elasticnet__l1_ratio": [
        0.1, 0.3, 0.5, 0.7, 0.9, 1.0
    ]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    model,
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1
)

print("正在训练模型...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# 9. 模型预测
y_pred = best_model.predict(X_test)

# 10. 模型评价
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== 模型评价 ==========")
print("最佳参数：", grid_search.best_params_)
print("交叉验证 RMSE：", round(-grid_search.best_score_, 4))
print("测试集 RMSE：", round(rmse, 4))
print("测试集 R²：", round(r2, 4))

# 11. 特征系数分析
feature_names = (
    best_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    best_model.named_steps["elasticnet"].coef_
)

coef_df = pd.DataFrame({
    "特征": feature_names,
    "回归系数": coefficients,
    "系数绝对值": np.abs(coefficients)
}).sort_values(
    by="系数绝对值",
    ascending=False
)

print("\n========== 重要特征 Top 20 ==========")
print(coef_df.head(20).to_string(index=False))

print(
    "\n非零系数数量：",
    int(np.sum(np.abs(coefficients) > 1e-8))
)

# 12. 实际值与预测值对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Release Year")
plt.ylabel("Predicted Release Year")
plt.title("Elastic Net: Actual vs Predicted")
plt.tight_layout()
plt.savefig("netflix_prediction.png", dpi=300)
plt.show()

# 13. 特征系数图
top_features = coef_df.head(20).sort_values(
    by="回归系数"
)

plt.figure(figsize=(10, 8))
plt.barh(
    top_features["特征"],
    top_features["回归系数"]
)

plt.xlabel("Regression Coefficient")
plt.title("Top 20 Elastic Net Feature Coefficients")
plt.tight_layout()
plt.savefig("netflix_feature_coefficients.png", dpi=300)
plt.show()

# 14. 输出预测结果
result_df = pd.DataFrame({
    "实际发行年份": y_test.to_numpy(),
    "预测发行年份": np.round(y_pred, 1),
    "预测误差": np.round(y_test.to_numpy() - y_pred, 1)
})

print("\n========== 部分预测结果 ==========")
print(result_df.head(10).to_string(index=False))

# 15. 保存结果
result_df.to_csv(
    "netflix_prediction_results.csv",
    index=False,
    encoding="utf-8-sig"
)

coef_df.to_csv(
    "netflix_feature_coefficients.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n分析完成！")
```
