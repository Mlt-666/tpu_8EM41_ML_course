import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("results/xgboost", exist_ok=True)

# 加载验证数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv")

# 加载模型
model = XGBRegressor()
model.load_model("models/xgboost_model.json")

# 预测
y_pred = model.predict(X_val)

# 评估指标
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# 保存指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/xgboost/metrics_valid.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 图像绘制
y_val_array = y_val.values.flatten()
y_pred_array = y_pred.flatten()

# 📈 实际 vs 预测图
max_val = max(y_val_array.max(), y_pred_array.max())
min_val = min(y_val_array.min(), y_pred_array.min())

plt.figure(figsize=(8, 6))
plt.scatter(y_val_array, y_pred_array, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("XGBoost: Actual vs Predicted (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/xgboost/actual_vs_pred_valid.png")
plt.close()

# 📉 残差图
residuals = y_val_array - y_pred_array
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_array, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("XGBoost: Residual Plot (Validation Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/xgboost/residual_plot_valid.png")
plt.close()

print(" XGBoost 验证阶段完成，图像与指标已输出")
