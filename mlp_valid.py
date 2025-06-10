# mlp_valid.py
import os
import json
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model

# —— 1. 创建输出目录 ——————————————————————————————————————————————————————————
os.makedirs("results/mlp", exist_ok=True)

# —— 2. 读取超参数 ————————————————————————————————————————————————————————————
if os.path.exists("params.yaml"):
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    print("当前 MLP 超参数：", params.get("mlp", {}))
else:
    print("⚠️ 未检测到 params.yaml，跳过超参数读取。")

# —— 3. 读取验证集 ——————————————————————————————————————————————————————————————
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").values.ravel()

# —— 4. 加载 scaler 并对验证集标准化 ————————————————————————————————————————
scaler_path = "models/mlp_scaler.pkl"
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"未找到 scaler 文件：{scaler_path}，请先运行训练脚本。")

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
X_val_scaled = scaler.transform(X_val)

# —— 5. 加载 Keras 模型 ——————————————————————————————————————————————————————————
model_path = "models/mlp_model.keras"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到模型文件：{model_path}，请先运行训练脚本。")

model = load_model(model_path)

# —— 6. 验证集预测与指标计算 ————————————————————————————————————————————————
y_pred = model.predict(X_val_scaled).flatten()

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2  = r2_score(y_val, y_pred)

metrics_valid = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}

# —— 7. 保存验证指标 ——————————————————————————————————————————————————————————
with open("results/mlp/metrics_valid.json", "w", encoding="utf-8") as f:
    json.dump(metrics_valid, f, indent=4, ensure_ascii=False)

print("✅ MLP 验证完成，评估指标已保存至 results/mlp/metrics_valid.json")
