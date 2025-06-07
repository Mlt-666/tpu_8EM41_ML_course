# mlp_valid.py

import os
import json
import joblib
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# —— 1. 创建输出目录 ——————————————————————————————————————————————————————————
os.makedirs("results/mlp", exist_ok=True)

# —— 2. （可选）打印当前 MLP 超参数 ——————————————————————————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    print("当前 MLP 超参数：", params.get("mlp", {}))
else:
    print("警告：未检测到 params.yaml，无法打印 MLP 超参数。")

# —— 3. 读取验证数据 ——————————————————————————————————————————————————————————
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").values.ravel()

# —— 4. 加载并使用训练时保存的 scaler 对验证特征做标准化 ——————————————————————
scaler_path = "models/mlp_scaler.pkl"
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    X_val_scaled = scaler.transform(X_val)
else:
    raise FileNotFoundError(f"未找到 scaler 文件：{scaler_path}，请先运行训练脚本并保存 scaler。")

# —— 5. 加载训练好的 MLP 模型 ————————————————————————————————————————————————————
model_path = "models/mlp_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"未找到模型文件：{model_path}")

# —— 6. 在验证集上预测并计算评估指标 ———————————————————————————————————————
y_pred = model.predict(X_val_scaled)
mse    = mean_squared_error(y_val, y_pred)
mae    = mean_absolute_error(y_val, y_pred)
r2     = r2_score(y_val, y_pred)

# —— 7. 保存验证集评估指标到 JSON ——————————————————————————————————————————
metrics_valid = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/mlp/metrics_valid.json", "w", encoding="utf-8") as f:
    json.dump(metrics_valid, f, indent=4)

# —— 8. 验证集可视化（实际 vs 预测 + 残差图）—————————————————————————————————————
# 8.1 实际 vs 预测 图
min_val = min(y_val.min(), y_pred.min())
max_val = max(y_val.max(), y_pred.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("MLP: Actual vs Predicted (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mlp/actual_vs_pred_valid.png")
plt.close()

# 8.2 残差图
residuals = y_val - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("MLP: Residual Plot (Validation Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mlp/residual_plot_valid.png")
plt.close()

print("✅ MLP 验证阶段完成：")
print("  • 验证集指标已保存至 results/mlp/metrics_valid.json")
print("  • 验证集可视化图已保存至 results/mlp/*.png")
