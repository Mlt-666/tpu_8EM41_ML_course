import os
import json
import joblib
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

# —— 1. 读取超参数 ———————————————————————————————
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

mlp_params = params.get("mlp", {})
hidden_layer_sizes = tuple(mlp_params.get("hidden_layer_sizes", [100, 50]))
activation         = mlp_params.get("activation", "relu")
solver             = mlp_params.get("solver", "adam")
max_iter           = mlp_params.get("max_iter", 1000)
random_state       = mlp_params.get("random_state", 42)
alpha              = mlp_params.get("alpha", 0.0001)

print("当前 MLP 超参数：")
print(f"  hidden_layer_sizes: {hidden_layer_sizes}")
print(f"  activation:         {activation}")
print(f"  solver:             {solver}")
print(f"  max_iter:           {max_iter}")
print(f"  random_state:       {random_state}")
print(f"  alpha (L2 penalty): {alpha}")

# —— 2. 创建目录 ——————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/mlp", exist_ok=True)
os.makedirs("runs/mlp", exist_ok=True)

# —— 3. TensorBoard ——————————————————————————————
writer = SummaryWriter(log_dir="runs/mlp")

# —— 4. 加载训练与验证数据 ————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv").values.ravel()

# —— 5. 标准化处理 ————————————————————————————
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
joblib.dump(scaler, "models/mlp_scaler.pkl")

# —— 6. 初始化并训练模型（带 early_stopping） ——
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    max_iter=max_iter,
    random_state=random_state,
    alpha=alpha,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    verbose=True
)
model.fit(X_train_scaled, y_train)

# —— 7. 模型评估 ————————————————————————————————
y_pred_train = model.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred_train)
mae = mean_absolute_error(y_train, y_pred_train)
r2 = r2_score(y_train, y_pred_train)

metrics_learn = {"R2": r2, "MAE": mae, "MSE": mse}
with open("results/mlp/metrics_learn.json", "w", encoding="utf-8") as f:
    json.dump(metrics_learn, f, indent=4)

joblib.dump(model, "models/mlp_model.pkl")

# —— 8. 绘制学习曲线（loss）与 MAE 曲线 ————————
train_mae, val_mae = [], []
for i in range(len(model.loss_curve_)):
    train_mae.append(mean_absolute_error(y_train, model.predict(X_train_scaled)))
    val_mae.append(mean_absolute_error(y_val, model.predict(X_val_scaled)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(model.loss_curve_, label="Train")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_mae, label="Train")
plt.plot(val_mae, label="Validation")
plt.title("Model MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("results/mlp/train_val_curves.png")
plt.close()

# —— 9. 权重直方图 ——————————————————————————————
for i, coef in enumerate(model.coefs_, 1):
    weights = coef.flatten()
    plt.figure(figsize=(8, 6))
    plt.hist(weights, bins=30, edgecolor="black")
    plt.title(f"Weight Histogram of Layer {i}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"results/mlp/weights_histogram_layer{i}.png")
    plt.close()
    writer.add_histogram(f"Layer{i}_weights", weights, 0)

# —— 10. 写入 TensorBoard ————————————————————————
writer.add_scalar("metrics/train_R2", r2, 0)
writer.add_scalar("metrics/train_MAE", mae, 0)
writer.add_scalar("metrics/train_MSE", mse, 0)
writer.close()

print("✅ 最终版 MLP 模型训练完成")
print("  • 模型已保存：models/mlp_model.pkl")
print("  • 指标写入：results/mlp/metrics_learn.json")
print("  • 曲线图保存：results/mlp/train_val_curves.png")
print("  • 权重图写入：results/mlp/weights_histogram_layer*.png")
