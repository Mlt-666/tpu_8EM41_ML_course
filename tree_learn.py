import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import yaml

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建输出目录 —————————————————————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/tree", exist_ok=True)

# —— 2. 从 params.yaml 中读取 max_depth 和 random_state ——————————————————————
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

tree_params = params.get("tree", {})
max_depth    = tree_params.get("max_depth", None)
random_state = tree_params.get("random_state", None)

if max_depth is None or random_state is None:
    raise KeyError(
        "请在 params.yaml 的 'tree' 段落里配置 'max_depth' 和 'random_state'。"
    )

# —— 3. 读取训练数据 ————————————————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 4. 初始化并训练决策树模型 ————————————————————————————————————————————————
model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
model.fit(X_train, y_train)

# —— 5. 在训练集上预测并计算指标 —————————————————————————————————————————————
y_pred = model.predict(X_train)
metrics_learn = {
    "R2": r2_score(y_train, y_pred),
    "MAE": mean_absolute_error(y_train, y_pred),
    "MSE": mean_squared_error(y_train, y_pred)
}

# —— 6. 保存训练集评估指标 ————————————————————————————————————————————————————
with open("results/tree/metrics_learn.json", "w") as f:
    json.dump(metrics_learn, f, indent=4)

# —— 7. 保存模型到 models/tree_model.pkl ——————————————————————————————————————————
joblib.dump(model, "models/tree_model.pkl")

# —— 8. 绘制并保存决策树结构的可视化（只显示前 3 层） ————————————————————————


plt.figure(figsize=(14, 8))  # 减小纵向空间，提高密度

plot_tree(
    model,
    feature_names=X_train.columns,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=10,         # 增大字体
    proportion=True      # 使用样本比例显示，避免太长的绝对数值
)

plt.title(f"Decision Tree (max_depth={max_depth}) - Top 3 Levels", fontsize=14)
plt.tight_layout()       # 自动紧凑排版，减少空白
plt.savefig("results/tree/tree_structure.svg", format='svg')
plt.close()


print("✅ 决策树训练完成并已保存：")
print(f"   • 模型文件：models/tree_model.pkl")
print(f"   • 训练集指标：results/tree/metrics_learn.json")
print(f"   • 结构图：     results/tree/tree_structure.png")
