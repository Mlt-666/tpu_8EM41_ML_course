# xgboost_learn.py

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# —— 1. 读取超参数 —————————————————————————————————————————————————
with open("params.yaml", "r",encoding="utf-8") as f:
    params = yaml.safe_load(f)

xgb_params = params.get("xgb", {})
learning_rate = xgb_params.get("learning_rate", 0.1)
max_depth      = xgb_params.get("max_depth", 6)
n_estimators   = xgb_params.get("n_estimators", 100)
random_state   = xgb_params.get("random_state", 42)

# —— 2. 创建输出目录 ——————————————————————————————————————————————————————
os.makedirs("models", exist_ok=True)
os.makedirs("results/xgboost", exist_ok=True)

# —— 3. 读取训练数据 ———————————————————————————————————————————————————————
X_train = pd.read_csv("prepare/X_train.csv")
y_train = pd.read_csv("prepare/y_train.csv").values.ravel()

# —— 4. 初始化并训练 XGBoost 模型 —————————————————————————————————————————————
model = XGBRegressor(
    learning_rate=learning_rate,
    max_depth=max_depth,
    n_estimators=n_estimators,
    random_state=random_state,
    use_label_encoder=False,
    eval_metric="rmse"
)
model.fit(X_train, y_train)

# —— 5. 在训练集上预测并计算评估指标 ———————————————————————————————————————
y_pred = model.predict(X_train)
mse    = mean_squared_error(y_train, y_pred)
mae    = mean_absolute_error(y_train, y_pred)
r2     = r2_score(y_train, y_pred)

# —— 6. 保存模型（JSON 格式） ——————————————————————————————————————————————
model.save_model("models/xgboost_model.json")

# —— 7. 保存训练集评估指标 ———————————————————————————————————————————————
metrics_learn = {
    "R2": r2,
    "MAE": mae,
    "MSE": mse
}
with open("results/xgboost/metrics_learn.json", "w") as f:
    json.dump(metrics_learn, f, indent=4)

# —— 8. 提取并保存特征重要性（CSV 和 PNG） —————————————————————————————————————
feature_importance = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
feat_df.to_csv("results/xgboost/feature_importance.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"])
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("results/xgboost/feature_importance.png")
plt.close()

print("✅ XGBoost 训练完成并已保存：")
print(f"   • 模型文件：models/xgboost_model.json")
print(f"   • 训练集指标：results/xgboost/metrics_learn.json")
print(f"   • 特征重要性 CSV：results/xgboost/feature_importance.csv")
print(f"   • 特征重要性图：results/xgboost/feature_importance.png")
