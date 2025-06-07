# catboost_learn.py

import os
import json
import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 读取超参数 —————————————————————————————————————————————————
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

cat_params = params.get("catboost", {})
# 如果 params.yaml 中缺少某个字段，就使用以下默认值：
depth         = cat_params.get("depth", 6)
iterations    = cat_params.get("iterations", 100)
learning_rate = cat_params.get("learning_rate", 0.1)
random_state  = cat_params.get("random_state", 42)

# —— 2. 创建输出目录 ————————————————————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/catboost", exist_ok=True)

# —— 3. 读取训练数据 ————————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 4. 初始化并训练 CatBoost 模型 ————————————————————————————————————————
model = CatBoostRegressor(
    depth=depth,
    learning_rate=learning_rate,
    iterations=iterations,
    random_state=random_state,
    verbose=0
)
model.fit(X_train, y_train)

# —— 5. 在训练集上做预测并计算评估指标 —————————————————————————————————————
y_pred = model.predict(X_train)
mse    = mean_squared_error(y_train, y_pred)
mae    = mean_absolute_error(y_train, y_pred)
r2     = r2_score(y_train, y_pred)

# —— 6. 保存模型 ————————————————————————————————————————————————
model.save_model("models/catboost_model.cbm")

# —— 7. 保存训练集评估指标 ————————————————————————————————————————————————
metrics_learn = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/catboost/metrics_learn.json", "w") as f:
    json.dump(metrics_learn, f, indent=4)

# —— 8. 保存特征重要性到 CSV & PNG ————————————————————————————————————————
feature_importances = model.get_feature_importance(prettified=False)
importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)
importance_df.to_csv("results/catboost/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("CatBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/catboost/feature_importance.png")
plt.close()

print("✅ CatBoost 训练完成并已保存：")
print(f"   • 模型：models/catboost_model.cbm")
print(f"   • 训练集指标：results/catboost/metrics_learn.json")
print(f"   • 特征重要性 CSV：results/catboost/feature_importance.csv")
print(f"   • 特征重要性图：results/catboost/feature_importance.png")
