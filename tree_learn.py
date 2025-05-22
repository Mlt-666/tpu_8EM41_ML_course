import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results/tree", exist_ok=True)

# 读取训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 初始化决策树模型（限制深度以便可视化）
model = DecisionTreeRegressor(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存模型
joblib.dump(model, "models/tree_model.pkl")

# 保存评估指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/tree/metrics_learn.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 可视化决策树前几层
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True, max_depth=3)
plt.title("Decision Tree: Top Levels")
plt.savefig("results/tree/tree_structure.png")
plt.close()

print(" 模型已保存至 models/tree_model.pkl")
print(" 指标已保存至 results/tree/metrics_learn.json")
print(" 决策树结构图已保存至 results/tree/tree_structure.png")
