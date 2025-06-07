# cv_mlp_search.py

import os
import yaml
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score

# —— 1. 加载训练数据 ——————————————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 2. 读取当前 mlp 默认超参数 —————————————————————————————————————————————
with open("params.yaml", "r", encoding="utf-8") as f:
    all_params = yaml.safe_load(f) or {}
mlp_defaults = all_params.get("mlp", {})
activation   = mlp_defaults.get("activation", "relu")
solver       = mlp_defaults.get("solver", "adam")
random_state = mlp_defaults.get("random_state", 42)

# —— 3. 构造 Pipeline ——————————————————————————————————————————————————————
pipe = Pipeline([
    ("scaler", StandardScaler()),       # 特征标准化
    ("mlp", MLPRegressor(                # MLP 模型
        activation=activation,
        solver=solver,
        random_state=random_state,
        verbose=False
    ))
])

# —— 4. 定义超参数网格 —— 只检索 hidden_layer_sizes 和 max_iter —————————————————————
param_grid = {
    "mlp__hidden_layer_sizes": [
        (50,), (100,), (100, 50),
        (150, 75), (200, 100, 50)
    ],
    "mlp__max_iter": [200, 400, 600]
}

# —— 5. 设置 5 折交叉验证与评分 ——————————————————————————————————————————————
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scorer = make_scorer(r2_score)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# —— 6. 运行网格搜索 ——————————————————————————————————————————————————————
grid.fit(X_train, y_train)

# —— 7. 输出最佳结果 ——————————————————————————————————————————————————————
print("最佳交叉验证 R²：", grid.best_score_)
print("最佳超参数：")
for k, v in grid.best_params_.items():
    print(f"  {k}: {v}")

# —— 8. 写回 params.yaml ——————————————————————————————————————————————————————
best_hidden = list(grid.best_params_["mlp__hidden_layer_sizes"])
best_max_it = int(grid.best_params_["mlp__max_iter"])

all_params["mlp"] = {
    "activation": activation,
    "hidden_layer_sizes": best_hidden,
    "max_iter": best_max_it,
    "random_state": random_state,
    "solver": solver
}

with open("params.yaml", "w", encoding="utf-8") as f:
    yaml.dump(all_params, f, allow_unicode=True)

print("✅ 已将最优超参数写回 params.yaml")
print(f"  hidden_layer_sizes: {best_hidden}")
print(f"  max_iter:           {best_max_it}")
