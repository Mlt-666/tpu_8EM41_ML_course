import pandas as pd
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建必要目录
os.makedirs("models", exist_ok=True)
os.makedirs("results/linear", exist_ok=True)

# 读取数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存模型
joblib.dump(model, "models/linear_model.pkl")

# 保存训练指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/linear/metrics_learn.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(" 模型已保存至 models/linear_model.pkl")
print(" 指标已保存至 results/linear/metrics_learn.json")
