import pandas as pd
import joblib
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("results/linear", exist_ok=True)

# 读取验证数据
X_val = pd.read_csv("prepare/X_val.csv")
y_val = pd.read_csv("prepare/y_val.csv")

# 加载训练好的模型
model = joblib.load("models/linear_model.pkl")

# 进行预测
y_pred = model.predict(X_val)

# 评估指标
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

# 保存验证集评估指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/linear/metrics_valid.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(" 验证集评估完成，结果已保存至 results/linear/metrics_valid.json")
