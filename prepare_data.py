import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 数据处理（确保已排除异常值、填补缺失）
housing_data = pd.read_csv("housing.csv")
housing_data = housing_data[housing_data["median_house_value"] < 500001]
housing_data["total_bedrooms"] = housing_data["total_bedrooms"].fillna(housing_data["total_bedrooms"].median())

# One-Hot 编码 ocean_proximity
housing_data_encoded = pd.get_dummies(housing_data, columns=["ocean_proximity"], drop_first=True)

# 最终选用特征
selected_features = [
    'median_income',
    'ocean_proximity_NEAR BAY',
    'total_rooms',
    'ocean_proximity_NEAR OCEAN',
    'latitude',
    'ocean_proximity_INLAND'
]

X = housing_data_encoded[selected_features]
y = housing_data_encoded["median_house_value"]

# 拆分数据（60%训练，40%验证）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# 确保 prepare 目录存在
save_dir = "prepare"
os.makedirs(save_dir, exist_ok=True)

# 保存文件
X_train.to_csv(f"{save_dir}/X_train.csv", index=False)
y_train.to_csv(f"{save_dir}/y_train.csv", index=False)
X_val.to_csv(f"{save_dir}/X_val.csv", index=False)
y_val.to_csv(f"{save_dir}/y_val.csv", index=False)
X.to_csv(f"{save_dir}/X_full.csv", index=False)
y.to_csv(f"{save_dir}/y_full.csv", index=False)

print(" 所有数据文件已保存至 prepare/")
