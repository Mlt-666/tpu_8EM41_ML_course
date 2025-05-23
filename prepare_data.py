import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 读取原始数据
df = pd.read_csv("housing.csv")

# 过滤异常值（只保留 < 500001）
df = df[df["median_house_value"] < 500001]

# One-hot 编码 ocean_proximity
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# 特征与目标
X = df.drop("median_house_value", axis=1)
y = df[["median_house_value"]]

# 划分训练集/验证集（60% / 40%）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# 创建输出文件夹
os.makedirs("prepare", exist_ok=True)

# 保存全部数据
X_train.to_csv("prepare/X_train.csv", index=False)
y_train.to_csv("prepare/y_train.csv", index=False)
X_val.to_csv("prepare/X_val.csv", index=False)
y_val.to_csv("prepare/y_val.csv", index=False)
X.to_csv("prepare/X_full.csv", index=False)
y.to_csv("prepare/y_full.csv", index=False)

print(" 数据已划分并保存至 prepare/")
