import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建结果目录
os.makedirs("results/catboost", exist_ok=True)

# 加载全数据
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv")

# 加载模型
model = CatBoostRegressor()
model.load_model("models/catboost_model.cbm")

# 预测
y_pred = model.predict(X_full)

# 指标计算
mse = mean_squared_error(y_full, y_pred)
mae = mean_absolute_error(y_full, y_pred)
r2 = r2_score(y_full, y_pred)

# 保存指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/catboost/metrics_full.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 图像输出
y_full_array = y_full.values.flatten()
y_pred_array = y_pred.flatten()

# 📈 实际 vs 预测图
max_val = max(y_full_array.max(), y_pred_array.max())
min_val = min(y_full_array.min(), y_pred_array.min())

plt.figure(figsize=(8, 6))
plt.scatter(y_full_array, y_pred_array, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("CatBoost: Actual vs Predicted (Full Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/actual_vs_pred_full.png")
plt.close()

# 📉 残差图
residuals = y_full_array - y_pred_array
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_array, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("CatBoost: Residual Plot (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/catboost/residual_plot_full.png")
plt.close()

print(" CatBoost 全数据评估完成，结果已保存至 results/catboost/")
