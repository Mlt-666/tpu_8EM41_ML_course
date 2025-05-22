import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results/xgboost", exist_ok=True)

# 加载训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 创建 XGBoost 模型
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测与评估
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存模型（JSON 格式）
model.save_model("models/xgboost_model.json")

# 保存评估指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/xgboost/metrics_learn.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 特征重要性提取
feature_importance = model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

# 保存为 CSV
importance_df.to_csv("results/xgboost/feature_importance.csv", index=False)

# 可视化并保存图
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/xgboost/feature_importance.png")
plt.close()

print(" XGBoost 模型已训练并保存")
print(" 指标写入 results/xgboost/metrics_learn.json")
print(" 特征重要性图 & CSV 保存成功")
