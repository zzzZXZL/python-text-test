import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ISLP import load_data
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


# ==================== 1. 加载数据集 ====================
datasets = {
    "Hitters": (load_data("Hitters").dropna(subset=["Salary"]), "Salary"),
    "Boston": (load_data("Boston"), "medv")
}

alphas = np.logspace(-3, 3, 40)
l1_ratios = [0.1, 0.5, 0.7, 0.9, 1.0]

cv = KFold(n_splits=10, shuffle=True, random_state=42)


# ==================== 2. 数据预处理 ====================
def prepare_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(exclude=np.number).columns

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols)
    ]

    if len(cat_cols) > 0:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers)

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return X_train, X_test, y_train, y_test, feature_names


# ==================== 3. 1-SE 规则 ====================
def one_se_rule(X, y, model_name):
    candidates = []

    if model_name == "Ridge":
        params = [(a, None) for a in alphas]
    elif model_name == "Lasso":
        params = [(a, None) for a in alphas]
    else:
        params = [(a, r) for r in l1_ratios for a in alphas]

    for alpha, ratio in params:
        if model_name == "Ridge":
            model = Ridge(alpha=alpha)
        elif model_name == "Lasso":
            model = Lasso(alpha=alpha, max_iter=30000)
        else:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=ratio,
                max_iter=30000
            )

        scores = -cross_val_score(
            model, X, y,
            cv=cv,
            scoring="neg_root_mean_squared_error"
        )

        candidates.append({
            "alpha": alpha,
            "ratio": ratio,
            "mean": scores.mean(),
            "se": scores.std(ddof=1) / np.sqrt(len(scores))
        })

    best = min(candidates, key=lambda item: item["mean"])
    threshold = best["mean"] + best["se"]

    eligible = [
        item for item in candidates
        if item["mean"] <= threshold
    ]

    if model_name == "ElasticNet":
        selected = max(
            eligible,
            key=lambda item: (item["ratio"], item["alpha"])
        )
    else:
        selected = max(
            eligible,
            key=lambda item: item["alpha"]
        )

    return best, selected


# ==================== 4. 模型比较 ====================
for dataset_name, (df, target) in datasets.items():

    print("\n" + "=" * 65)
    print("数据集：", dataset_name)

    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        df, target
    )

    # 统一使用10折交叉验证
    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1, max_iter=30000),
        "Elastic Net": ElasticNet(
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=30000
        )
    }

    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, model) in zip(axes, models.items()):

        # 交叉验证选择参数
        if name == "Ridge":
            from sklearn.linear_model import RidgeCV
            model = RidgeCV(alphas=alphas, cv=cv)

        elif name == "Lasso":
            from sklearn.linear_model import LassoCV
            model = LassoCV(
                alphas=alphas,
                cv=cv,
                max_iter=30000
            )

        else:
            from sklearn.linear_model import ElasticNetCV
            model = ElasticNetCV(
                alphas=alphas,
                l1_ratio=l1_ratios,
                cv=cv,
                max_iter=30000
            )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        nonzero = np.sum(np.abs(model.coef_) > 1e-8)

        results.append({
            "模型": name,
            "测试集RMSE": rmse,
            "非零系数数量": nonzero,
            "最优alpha": model.alpha_
        })

        # 输出被选择的变量
        selected_features = feature_names[
            np.abs(model.coef_) > 1e-8
        ]

        print(f"\n{name}：")
        print("最优 alpha：", model.alpha_)
        print("测试集 RMSE：", round(rmse, 4))
        print("非零变量数量：", nonzero)
        print("非零变量：", list(selected_features))

        # 绘制系数路径图
        for alpha in alphas:

            if name == "Ridge":
                path_model = Ridge(alpha=alpha)

            elif name == "Lasso":
                path_model = Lasso(
                    alpha=alpha,
                    max_iter=30000
                )

            else:
                path_model = ElasticNet(
                    alpha=alpha,
                    l1_ratio=model.l1_ratio_,
                    max_iter=30000
                )

            path_model.fit(X_train, y_train)

            ax.plot(
                alphas,
                path_model.coef_,
                linewidth=0.8
            )

        ax.set_xscale("log")
        ax.set_xlabel("Alpha (log scale)")
        ax.set_ylabel("Coefficient")
        ax.set_title(name + " Coefficient Paths")
        ax.grid(True, alpha=0.3)

    plt.suptitle(dataset_name + "：系数路径图")
    plt.tight_layout()
    plt.show()

    # 输出模型比较结果
    print("\n模型比较结果：")
    print(
        pd.DataFrame(results).round(4).to_string(index=False)
    )

    # ==================== 5. 1-SE 规则分析 ====================
    print("\n1-SE 规则分析：")

    for name in ["Ridge", "Lasso", "ElasticNet"]:

        best, selected = one_se_rule(
            X_train, y_train, name
        )

        print(f"\n{name}：")
        print(
            f"最小交叉验证 RMSE：{best['mean']:.4f}"
        )
        print(
            f"最优 alpha：{best['alpha']:.6f}"
        )
        print(
            f"1-SE 选择的 RMSE：{selected['mean']:.4f}"
        )
        print(
            f"1-SE 选择的 alpha：{selected['alpha']:.6f}"
        )

        if selected["ratio"] is not None:
            print(
                f"1-SE 选择的 l1_ratio：{selected['ratio']}"
            )

print("\n所有实验完成！")