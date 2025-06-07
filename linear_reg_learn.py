# linear_reg_learn.py

import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建输出目录 —————————————————————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/linear", exist_ok=True)

# —— 2. 读取超参数 —— 仅提取 normalize 并移除 ——————————————————————————————
with open("params.yaml", "r") as f:
    params_all = yaml.safe_load(f)

linear_params = params_all.get("linear", {}).copy()
if "normalize" in linear_params:
    linear_params.pop("normalize")

# —— 3. 读取训练数据 ——————————————————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# —— 4. 初始化并训练模型 ————————————————————————————————————————————————————————
# 此时 linear_params 是空 dict，相当于调用 LinearRegression()
model = LinearRegression(**linear_params)
model.fit(X_train, y_train)

# —— 5. 在训练集上预测并计算指标 ————————————————————————————————————————————————
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2  = r2_score(y_train, y_pred)

# —— 6. 保存模型 ————————————————————————————————————————————————————————————————
joblib.dump(model, "models/linear_model.pkl")

# —— 7. 保存训练集上的评估指标 ————————————————————————————————————————————————
metrics_learn = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/linear/metrics_learn.json", "w") as f:
    json.dump(metrics_learn, f, indent=4)

# —— 8. 训练集可视化（实际 vs 预测 + 残差图） ——————————————————————————————————————————
y_train_arr = y_train.values.flatten()
y_pred_arr  = y_pred.flatten()

# —— 8.1 实际 vs 预测（带理想拟合线） ————————————————————————————————————————————
min_val = min(y_train_arr.min(), y_pred_arr.min())
max_val = max(y_train_arr.max(), y_pred_arr.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_train_arr, y_pred_arr, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted (Train Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear/actual_vs_pred.png")
plt.close()

# —— 8.2 残差图 ——————————————————————————————————————————————————————————————
residuals = y_train_arr - y_pred_arr
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Train Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear/residual_plot.png")
plt.close()

# —— 9. 保存模型系数与截距 ————————————————————————————————————————————————————————
weights = {
    "coef": model.coef_.tolist(),
    "intercept": float(model.intercept_) if hasattr(model.intercept_, "item") else model.intercept_
}
with open("results/linear/weights.json", "w") as f:
    json.dump(weights, f, indent=4)

# —— 10. 输出完成信息 ——————————————————————————————————————————————————————————
print("✔ 训练结束：")
print("  • 模型已保存至 models/linear_model.pkl")
print("  • 训练集指标已保存至 results/linear/metrics_learn.json")
print("  • 训练集可视化图已保存至 results/linear/*.png")
print("  • 权重信息已保存至 results/linear/weights.json")
