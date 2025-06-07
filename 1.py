import os
import json
import yaml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score

# —— 1. 创建输出目录 ——————————————————————————————
os.makedirs("results/mlp", exist_ok=True)

# —— 2. 加载训练数据 ———————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 3. 构建 Pipeline（标准化 + MLP）————————————————
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(max_iter=1000, random_state=42))
])

# —— 4. 定义超参数搜索空间 ————————————————————————
param_grid = {
    "mlp__hidden_layer_sizes": [(64, 32), (100, 50), (128, 64, 32)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__solver": ["adam"],
    "mlp__alpha": [0.0001, 0.001, 0.01],
}

# —— 5. 配置交叉验证（5折）并设置评分函数为 R² ——————
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring=make_scorer(r2_score),
    verbose=2,
    n_jobs=-1
)

# —— 6. 执行搜索 ————————————————————————————————
grid_search.fit(X_train, y_train)

# —— 7. 保存最佳模型参数和得分 ————————————————————
best_params = grid_search.best_params_
best_score = grid_search.best_score_

result = {
    "best_params": best_params,
    "best_cv_r2": best_score
}

with open("results/mlp/mlp_cv_results.json", "w") as f:
    json.dump(result, f, indent=4)

print("✅ 超参数搜索完成")
print("最佳 R²（CV）：", best_score)
print("最佳参数：", best_params)
