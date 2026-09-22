
# 导入需要的库
from ISLP import load_data
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# 1. 加载 Carseats 数据集
Carseats = load_data("Carseats")

# 查看数据
print(Carseats.head())
print("数据维度：", Carseats.shape)


# 2. 选择需要的变量
data = Carseats[
    ["Sales", "Price", "Income", "Advertising", "ShelveLoc"]
].copy()


# 3. 将 ShelveLoc 转换为哑变量
# drop_first=True 表示删除第一个类别作为基准组
# 因此 Bad 为基准组
data = pd.get_dummies(
    data,
    columns=["ShelveLoc"],
    drop_first=True
)

# 查看转换后的数据
print("\n转换后的数据：")
print(data.head())


# 4. 设置因变量 Y
y = data["Sales"]


# 5. 设置自变量 X
X = data[
    [
        "Price",
        "Income",
        "Advertising",
        "ShelveLoc_Good",
        "ShelveLoc_Medium"
    ]
]


# 6. 添加截距项
X = sm.add_constant(X)


# 7. 建立多元线性回归模型
model = sm.OLS(y, X).fit()


# 8. 输出回归结果
print("\n================ 回归结果 ================\n")
print(model.summary())


# 9. 输出 ShelveLoc 基准组
print("\nShelveLoc 的基准组：Bad")


# 10. 计算 VIF
vif = pd.DataFrame()

vif["Variable"] = X.columns

vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]


# 11. 输出 VIF 结果
print("\n================ VIF结果 ================\n")
print(vif)


# 12. 判断多重共线性
print("\n================ 多重共线性判断 ================\n")

for i in range(1, len(vif)):  # 跳过 const
    variable = vif.loc[i, "Variable"]
    value = vif.loc[i, "VIF"]

    if value > 10:
        print(f"{variable}: VIF = {value:.2f}，存在较严重的多重共线性")
    elif value > 5:
        print(f"{variable}: VIF = {value:.2f}，需要关注多重共线性")
    else:
        print(f"{variable}: VIF = {value:.2f}，不存在明显的多重共线性")