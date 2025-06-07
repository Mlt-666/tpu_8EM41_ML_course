# catboost_valid.py

import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建结果目录 ————————————————————————————————————————————————
os.makedirs("results/catboost", exist_ok=True)

# —— 2. 读取并打印当前超参数（可选，仅用于日志） ——————————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    print("当前 CatBoost 超参数：", params.get("catboost", {}))

# —— 3. 读取验证数据 ————————————————————————————————————————————————
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").values.ravel()

# —— 4. 加载训练好的模型 ————————————————————————————————————————————————
model = CatBoostRegressor()
model.load_model("models/catboost_model.cbm")

# —— 5. 在验证集上预测并计算评估指标 —————————————————————————————————————
y_pred = model.predict(X_val)
mse    = mean_squared_error(y_val, y_pred)
mae    = mean_absolute_error(y_val, y_pred)
r2     = r2_score(y_val, y_pred)

# —— 6. 保存验证集评估指标 ————————————————————————————————————————————————
metrics_valid = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/catboost/metrics_valid.json", "w") as f:
    json.dump(metrics_valid, f, indent=4)

# —— 7. 验证集可视化（实际 vs 预测 + 残差图） ——————————————————————————————————
y_val_arr    = y_val
y_pred_arr_v = y_pred

# 7.1 实际 vs 预测 图
min_val = min(y_val_arr.min(),   y_pred_arr_v.min())
max_val = max(y_val_arr.max(),   y_pred_arr_v.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_val_arr, y_pred_arr_v, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("CatBoost: Actual vs Predicted (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/actual_vs_pred_valid.png")
plt.close()

# 7.2 残差图
residuals = y_val_arr - y_pred_arr_v
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_v, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("CatBoost: Residual Plot (Validation Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/residual_plot_valid.png")
plt.close()

print("✅ CatBoost 验证完成：")
print(f"   • 验证集指标：results/catboost/metrics_valid.json")
print(f"   • 实际 vs 预测图：results/catboost/actual_vs_pred_valid.png")
print(f"   • 残差图：        results/catboost/residual_plot_valid.png")
