# linear_reg_valid.py

import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建输出目录 —————————————————————————————————————————————————
os.makedirs("results/linear", exist_ok=True)

# —— 2. 读取超参数 —— 仅示范，验证阶段不使用 normalize ——————————————————————————
with open("params.yaml", "r") as f:
    params_all = yaml.safe_load(f)
# linear_params = params_all.get("linear", {})  # 如果后续需要可视化参数可用它

# —— 3. 加载训练好的模型 —————————————————————————————————————————————————————
model = joblib.load("models/linear_model.pkl")

# —— 4. 读取验证数据 ——————————————————————————————————————————————————————————
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv")

# —— 5. 在验证集上预测并计算指标 ——————————————————————————————————————————————
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

# —— 6. 保存验证集评估指标 ————————————————————————————————————————————————
metrics_valid = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/linear/metrics_valid.json", "w") as f:
    json.dump(metrics_valid, f, indent=4)

# —— 7. 验证集可视化（实际 vs 预测 + 残差图） ——————————————————————————————————————————
y_val_arr    = y_val.values.flatten()
y_pred_arr_v = y_pred.flatten()

# —— 7.1 实际 vs 预测（验证集） ——————————————————————————————————————————————
min_val = min(y_val_arr.min(), y_pred_arr_v.min())
max_val = max(y_val_arr.max(), y_pred_arr_v.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_val_arr, y_pred_arr_v, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression: Actual vs Predicted (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear/actual_vs_pred_val.png")
plt.close()

# —— 7.2 残差图（验证集） ————————————————————————————————————————————————————
residuals = y_val_arr - y_pred_arr_v
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_v, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residual Plot (Validation Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/linear/residual_plot_val.png")
plt.close()

# —— 8. 输出完成信息 ——————————————————————————————————————————————————————————
print("✔ 验证结束：")
print("  • 验证集指标已保存至 results/linear/metrics_valid.json")
print("  • 验证集可视化图已保存至 results/linear/*.png")
