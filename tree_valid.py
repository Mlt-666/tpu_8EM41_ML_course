import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 创建结果目录 ——————————————————————————————————————————————————————————
os.makedirs("results/tree", exist_ok=True)

# —— 2. 读取当前 params.yaml 中的 tree 参数（仅用于日志打印） ————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    print("当前使用的决策树超参数：", params.get("tree", {}))

# —— 3. 读取验证数据 ——————————————————————————————————————————————————————————
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").values.ravel()

# —— 4. 加载已训练好的模型 —————————————————————————————————————————————————————
model = joblib.load("models/tree_model.pkl")

# —— 5. 在验证集上预测并计算指标 ——————————————————————————————————————————————
y_pred = model.predict(X_val)
metrics_valid = {
    "R2": r2_score(y_val, y_pred),
    "MAE": mean_absolute_error(y_val, y_pred),
    "MSE": mean_squared_error(y_val, y_pred)
}

# —— 6. 保存验证集评估指标 ————————————————————————————————————————————————————
with open("results/tree/metrics_valid.json", "w") as f:
    json.dump(metrics_valid, f, indent=4)

# —— 7. 验证集可视化（实际 vs 预测 + 残差图） ——————————————————————————————————————
y_val_arr    = y_val
y_pred_arr_v = y_pred

# 7.1 实际 vs 预测
min_val = min(y_val_arr.min(), y_pred_arr_v.min())
max_val = max(y_val_arr.max(), y_pred_arr_v.max())

plt.figure(figsize=(8, 6))
plt.scatter(y_val_arr, y_pred_arr_v, alpha=0.5, label="Prediction")
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Ideal Fit")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Decision Tree: Actual vs Predicted (Validation Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/actual_vs_pred_valid.png")
plt.close()

# 7.2 残差图
residuals = y_val_arr - y_pred_arr_v
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_arr_v, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Decision Tree: Residual Plot (Validation Set)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/tree/residual_plot_valid.png")
plt.close()

print("✅ 验证阶段完成：")
print("  • 验证集指标已保存至 results/tree/metrics_valid.json")
print("  • 验证集可视化图已保存至 results/tree/actual_vs_pred_valid.png")
print("  • 验证集残差图已保存至 results/tree/residual_plot_valid.png")
