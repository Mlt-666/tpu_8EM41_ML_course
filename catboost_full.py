# catboost_full_eval.py

import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建结果目录 ——————————————————————————————————————————————————————————
os.makedirs("results/catboost", exist_ok=True)

# —— 2. （可选）打印当前使用的 CatBoost 超参数 ————————————————————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    print("当前 CatBoost 超参数：", params.get("catboost", {}))
else:
    print("警告：未检测到 params.yaml，无法打印 CatBoost 超参数。")

# —— 3. 读取全量数据 ——————————————————————————————————————————————————————————
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv").values.ravel()

# —— 4. 加载已训练好的 CatBoost 模型 ——————————————————————————————————————————
model = CatBoostRegressor()
model.load_model("models/catboost_model.cbm")

# —— 5. 在全量数据上预测并计算评估指标 ————————————————————————————————————————
y_pred = model.predict(X_full)
mse    = mean_squared_error(y_full, y_pred)
mae    = mean_absolute_error(y_full, y_pred)
r2     = r2_score(y_full, y_pred)

# —— 6. 保存全量数据评估指标 ————————————————————————————————————————————————————
metrics_full = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/catboost/metrics_full.json", "w") as f:
    json.dump(metrics_full, f, indent=4)

# —— 7. 全量数据可视化：实际 vs 预测 + 残差图 ——————————————————————————————————————
y_full_arr   = y_full
y_pred_arr_f = y_pred

# 7.1 实际 vs 预测 图
min_val = min(y_full_arr.min(),   y_pred_arr_f.min())
max_val = max(y_full_arr.max(),   y_pred_arr_f.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_full_arr, y_pred_arr_f, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("CatBoost: Actual vs Predicted (Full Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/actual_vs_pred_full.png")
plt.close()

# 7.2 残差图
residuals_full = y_full_arr - y_pred_arr_f
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_f, residuals_full, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("CatBoost: Residual Plot (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/residual_plot_full.png")
plt.close()

print("✅ CatBoost 全数据评估完成：")
print("  • 全量数据指标已保存至 results/catboost/metrics_full.json")
print("  • 实际 vs 预测图已保存至 results/catboost/actual_vs_pred_full.png")
print("  • 残差图已保存至 results/catboost/residual_plot_full.png")
