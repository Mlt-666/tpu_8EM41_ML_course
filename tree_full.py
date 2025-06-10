# tree_full_eval.py

import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建输出目录 ——————————————————————————————————————————————————————————
os.makedirs("results/tree", exist_ok=True)

# —— 2. 读取当前 params.yaml 中的 'tree' 配置（仅用于日志打印） ————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    tree_params = params.get("tree", {})
    print("当前使用的决策树超参数：", tree_params)
else:
    tree_params = {}
    print("警告：未检测到 params.yaml，无法打印 'tree' 参数。")

# —— 3. 读取全量数据 ——————————————————————————————————————————————————————————
X_full = pd.read_csv("prepare/X_full.csv")
y_full = pd.read_csv("prepare/y_full.csv").values.ravel()

# —— 4. 加载已训练好的模型 —————————————————————————————————————————————————————
model = joblib.load("models/tree_model.pkl")

# —— 5. 在全量数据上预测并计算指标 ——————————————————————————————————————————————
y_pred = model.predict(X_full)
mse    = mean_squared_error(y_full, y_pred)
mae    = mean_absolute_error(y_full, y_pred)
r2     = r2_score(y_full, y_pred)

# —— 6. 保存全量数据评估指标 —— results/tree/metrics_full.json ——————————————————————
metrics_full = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/tree/metrics_full.json", "w") as f:
    json.dump(metrics_full, f, indent=4)

# —— 7. 绘制并保存可视化图 —— 实际 vs 预测 + 残差图 ——————————————————————————————
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
plt.title("Decision Tree: Actual vs Predicted (Full Data)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/actual_vs_pred_full.png")
plt.close()

# 7.2 残差图
residuals_full = y_full_arr - y_pred_arr_f
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_f, residuals_full, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Decision Tree: Residual Plot (Full Data)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/residual_plot_full.png")
plt.close()

print("✅ 决策树全量数据评估完成：")
print("  • 全量数据指标已保存至 results/tree/metrics_full.json")
print("  • 实际 vs 预测图已保存至 results/tree/actual_vs_pred_full.png")
print("  • 残差图已保存至 results/tree/residual_plot_full.png")
