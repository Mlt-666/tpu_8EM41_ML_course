import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("results/tree", exist_ok=True)

# 读取 full 数据
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv")

# 加载模型
model = joblib.load("models/tree_model.pkl")

# 预测与评估
y_pred = model.predict(X_full)
mse = mean_squared_error(y_full, y_pred)
mae = mean_absolute_error(y_full, y_pred)
r2 = r2_score(y_full, y_pred)

# 保存指标 JSON
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/tree/metrics_full.json", "w") as f:
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
plt.title("Decision Tree: Actual vs Predicted (Full Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/actual_vs_pred_full.png")
plt.close()

# 📉 残差图
residuals = y_full_array - y_pred_array
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_array, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Decision Tree: Residual Plot (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/residual_plot_full.png")
plt.close()

print(" 决策树全数据评估完成：指标与图像已保存到 results/tree/")
