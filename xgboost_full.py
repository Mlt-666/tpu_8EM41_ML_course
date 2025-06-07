# xgboost_full_eval.py

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建输出目录 ——————————————————————————————————————————————————————————
os.makedirs("results/xgboost", exist_ok=True)

# —— 2. （可选）打印当前使用的 XGBoost 超参数 ————————————————————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    xgb_params = params.get("xgb", {})
    print("当前 XGBoost 超参数：", xgb_params)
else:
    print("警告：未检测到 params.yaml，无法打印 XGBoost 超参数。")

# —— 3. 读取全量数据 ——————————————————————————————————————————————————————————
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv").values.ravel()

# —— 4. 加载已训练好的 XGBoost 模型 ——————————————————————————————————————————
model = XGBRegressor()
model.load_model("models/xgboost_model.json")

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
with open("results/xgboost/metrics_full.json", "w") as f:
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
plt.title("XGBoost: Actual vs Predicted (Full Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/xgboost/actual_vs_pred_full.png")
plt.close()

# 7.2 残差图
residuals_full = y_full_arr - y_pred_arr_f
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_f, residuals_full, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("XGBoost: Residual Plot (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/xgboost/residual_plot_full.png")
plt.close()

print("✅ XGBoost 全数据评估完成：")
print("  • 全量数据指标已保存至 results/xgboost/metrics_full.json")
print("  • 实际 vs 预测图已保存至 results/xgboost/actual_vs_pred_full.png")
print("  • 残差图已保存至 results/xgboost/residual_plot_full.png")
