import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# 📁 创建输出目录
output_dir = "results/compare_models"
os.makedirs(output_dir, exist_ok=True)

# 📂 模型指标文件路径
model_files = {
    "Linear Regression": "results/linear/metrics_full.json",
    "Decision Tree": "results/tree/metrics_full.json",
    "CatBoost": "results/catboost/metrics_full.json",
    "XGBoost": "results/xgboost/metrics_full.json",
    "MLP": "results/mlp/metrics_full.json"
}

# 📊 读取 JSON 指标
metrics_data = []
for model, path in model_files.items():
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            data["Model"] = model
            metrics_data.append(data)
    except:
        print(f" 无法读取: {path}")
        metrics_data.append({"Model": model, "R2": None, "MAE": None, "MSE": None})

# ➕ 整理为 DataFrame
df = pd.DataFrame(metrics_data)
df = df[["Model", "R2", "MAE", "MSE"]]
df = df.sort_values(by="R2", ascending=False)

# 📋 保存表格为 CSV（可选）
df.to_csv(f"{output_dir}/model_metrics_table.csv", index=False)

# ✅ 打印终端 Markdown 表格
print("\n 模型指标对比表：")
print(df.to_markdown(index=False))

# 📈 图像 1：R² 越高越好
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["R2"], color="mediumseagreen")
plt.title("模型对比：R² 越高越好")
plt.ylabel("R²")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_r2_comparison.png")
plt.close()

# 📉 图像 2：MAE 越低越好
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["MAE"], color="salmon")
plt.title("模型对比：MAE 越低越好")
plt.ylabel("MAE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_mae_comparison.png")
plt.close()

# 📉 图像 3：MSE 越低越好
plt.figure(figsize=(10, 5))
plt.bar(df["Model"], df["MSE"], color="cornflowerblue")
plt.title("模型对比：MSE 越低越好")
plt.ylabel("MSE")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/model_mse_comparison.png")
plt.close()

print("\n 所有图像已保存至: results/compare_models/")
