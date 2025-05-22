import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results/mlp", exist_ok=True)
os.makedirs("runs/mlp", exist_ok=True)

# TensorBoard 日志器
writer = SummaryWriter(log_dir="runs/mlp")

# 加载数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 创建并训练模型
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                     max_iter=500, random_state=42, verbose=False)
model.fit(X_train, y_train.values.ravel())

# 预测 & 指标
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存模型
joblib.dump(model, "models/mlp_model.pkl")

# 保存指标 JSON
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/mlp/metrics_learn.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 📈 学习曲线
plt.figure(figsize=(8, 6))
plt.plot(model.loss_curve_, label='Training Loss')
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("MLP Learning Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mlp/loss_curve.png")
plt.close()

# ⛓️ 权重直方图 + TensorBoard 权重记录
layer_idx = 1
for coef in model.coefs_:
    weights = coef.flatten()
    plt.figure(figsize=(8, 6))
    plt.hist(weights, bins=30, edgecolor='black')
    plt.title(f"Weight Histogram of Layer {layer_idx}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    hist_path = f"results/mlp/weights_histogram_layer{layer_idx}.png"
    plt.savefig(hist_path)
    plt.close()
    
    # 写入 TensorBoard
    writer.add_histogram(f"Layer{layer_idx}_weights", weights, 0)
    layer_idx += 1

# 添加指标到 TensorBoard
writer.add_scalar("metrics/R2", r2, 0)
writer.add_scalar("metrics/MAE", mae, 0)
writer.add_scalar("metrics/MSE", mse, 0)
writer.close()

print(" MLP 模型训练完成")
print(" 指标保存至 results/mlp/metrics_learn.json")
print(" 学习曲线 & 权重图已保存")
print(" TensorBoard 日志写入 runs/mlp/")
