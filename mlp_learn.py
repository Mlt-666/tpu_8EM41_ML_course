import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.tensorboard import SummaryWriter

# —— 1. 读取超参数（指定 UTF-8 编码）———————————————————————————————
with open("params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

mlp_params = params.get("mlp", {})
hidden_layer_sizes = tuple(mlp_params.get("hidden_layer_sizes", [100, 50]))
activation         = mlp_params.get("activation", "relu")
solver             = mlp_params.get("solver", "adam")
max_iter           = mlp_params.get("max_iter", 1000)
random_state       = mlp_params.get("random_state", 42)
alpha              = mlp_params.get("alpha", 0.0001)  # 加入 L2 正则项 alpha

print("当前 MLP 超参数：")
print(f"  hidden_layer_sizes: {hidden_layer_sizes}")
print(f"  activation:         {activation}")
print(f"  solver:             {solver}")
print(f"  max_iter:           {max_iter}")
print(f"  random_state:       {random_state}")
print(f"  alpha (L2 penalty): {alpha}")

# —— 2. 创建输出目录 ——————————————————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/mlp", exist_ok=True)
os.makedirs("runs/mlp", exist_ok=True)

# —— 3. TensorBoard 日志器 ————————————————————————————————————————
writer = SummaryWriter(log_dir="runs/mlp")

# —— 4. 加载训练数据 ——————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 5. 标准化处理 ————————————————————————————————————————————————
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, "models/mlp_scaler.pkl")

# —— 6. 初始化并训练模型 ——————————————————————————————————————————
model = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    solver=solver,
    max_iter=max_iter,
    random_state=random_state,
    alpha=alpha,  # 加入正则化
    verbose=False
)
model.fit(X_train_scaled, y_train)

# —— 7. 评估训练集性能 —————————————————————————————————————————————
y_pred_train = model.predict(X_train_scaled)
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

metrics_learn = {"R2": r2_train, "MAE": mae_train, "MSE": mse_train}
with open("results/mlp/metrics_learn.json", "w", encoding="utf-8") as f:
    json.dump(metrics_learn, f, indent=4)

joblib.dump(model, "models/mlp_model.pkl")

# —— 8. 绘制学习曲线 ————————————————————————————————————————————————
plt.figure(figsize=(8, 6))
plt.plot(model.loss_curve_)
plt.title("MLP Learning Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/mlp/loss_curve.png")
plt.close()

# —— 9. 绘制并保存权重直方图 ———————————————————————————————————————
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

# —— 10. 写入 TensorBoard 指标 ————————————————————————————————————————
writer.add_scalar("metrics/train_R2", r2_train, 0)
writer.add_scalar("metrics/train_MAE", mae_train, 0)
writer.add_scalar("metrics/train_MSE", mse_train, 0)
writer.close()

print("✅ MLP 模型训练完成")
print("  • 模型已保存：models/mlp_model.pkl")
print("  • 训练指标：results/mlp/metrics_learn.json")
print("  • 权重图与学习曲线已保存")
