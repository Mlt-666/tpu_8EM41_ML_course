import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 创建输出目录
os.makedirs("models", exist_ok=True)
os.makedirs("results/catboost", exist_ok=True)

# 读取训练数据
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv")

# 创建 CatBoost 模型
model = CatBoostRegressor(verbose=0, random_state=42)
model.fit(X_train, y_train)

# 预测 & 评估
y_pred = model.predict(X_train)
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)

# 保存模型
model.save_model("models/catboost_model.cbm")

# 保存指标
metrics = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/catboost/metrics_learn.json", "w") as f:
    json.dump(metrics, f, indent=4)

# 特征重要性提取
feature_importances = model.get_feature_importance(prettified=True)
feature_names = X_train.columns

# 保存为 CSV
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.get_feature_importance()
}).sort_values(by="Importance", ascending=False)

importance_df.to_csv("results/catboost/feature_importance.csv", index=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("CatBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/catboost/feature_importance.png")
plt.close()

print(" CatBoost 模型已训练并保存")
print(" 指标保存至 results/catboost/metrics_learn.json")
print(" 特征重要性保存至 results/catboost/feature_importance.[csv, png]")
