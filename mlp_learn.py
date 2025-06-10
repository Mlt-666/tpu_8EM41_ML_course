import os
import json
import yaml
import pickle
import math
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    print("开始 MLP (Keras) 模型训练...")

    # 1. 读取超参数
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    mlp_params = params.get("mlp", {})
    hidden_sizes = mlp_params.get("hidden_layer_sizes", [100, 50])
    activation   = mlp_params.get("activation", "relu")
    learning_rate = 1e-3  # 固定
    epochs        = mlp_params.get("max_iter", 1000)
    batch_size    = 32    # 固定
    patience      = 20    # 固定
    random_seed   = mlp_params.get("random_state", 42)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    print("Loaded mlp params:", mlp_params)

    # 2. 创建输出目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/mlp", exist_ok=True)
    run_logdir = os.path.join("runs", "mlp", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_logdir, exist_ok=True)

    # 3. TensorBoard & EarlyStopping 回调
    tb_cb = callbacks.TensorBoard(
        log_dir=run_logdir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch',
        profile_batch=0
    )
    es_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # 4. 加载并预处理训练数据
    X = pd.read_csv("prepare/X_train.csv")
    y = pd.read_csv("prepare/y_train.csv").values.ravel()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    with open("models/mlp_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # 5. 动态构建模型
    model = models.Sequential(name="mlp_regressor")
    model.add(layers.Input(shape=(X_scaled.shape[1],), name="input"))
    for units in hidden_sizes:
        # 每一隐藏层：Dense → BatchNorm
        model.add(layers.Dense(units, activation=activation, name=f"dense_{units}"))
        model.add(layers.BatchNormalization(name=f"bn_{units}"))
    # 输出层
    model.add(layers.Dense(1, activation="linear", name="output"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    model.summary()

    # 6. 训练
    history = model.fit(
        X_scaled, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[tb_cb, es_cb],
        verbose=1
    )

    # 7. 保存模型
    model.save("models/mlp_model.keras")
    print("✅ 模型已保存到 models/mlp_model.keras")

    # 8. 评估并保存指标
    y_pred = model.predict(X_scaled).flatten()
    mse   = mean_squared_error(y, y_pred)
    rmse  = math.sqrt(mse)
    mae   = mean_absolute_error(y, y_pred)
    r2    = r2_score(y, y_pred)

    metrics = {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "hidden_layer_sizes": hidden_sizes,
        "activation": activation,
        "max_iter (epochs)": epochs,
        "random_state": random_seed
    }
    with open("results/mlp/metrics_learn.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print("✅ 训练集评估指标已保存到 results/mlp/metrics_learn.json")

    # 9. 绘制学习曲线
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Model Loss"); plt.xlabel("Epoch"); plt.ylabel("MSE")
    plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(history.history["mae"], label="Train MAE")
    plt.plot(history.history["val_mae"], label="Val MAE")
    plt.title("Model MAE"); plt.xlabel("Epoch"); plt.ylabel("MAE")
    plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/mlp/mlp_learning_curves.png", dpi=300)
    plt.close()
    print("✅ 学习曲线已保存到 results/mlp/mlp_learning_curves.png")

    # 10. 绘制每层权重直方图（TensorBoard 也有）
    for layer in model.layers:
        if hasattr(layer, "kernel"):
            weights = layer.kernel.numpy().flatten()
            plt.figure(figsize=(6,4))
            plt.hist(weights, bins=30)
            plt.title(f"Weights of {layer.name}")
            plt.xlabel("Value"); plt.ylabel("Frequency"); plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/mlp/weights_{layer.name}.png")
            plt.close()
    print("✅ 权重直方图已保存到 results/mlp/")

    print(f"\n📊 TensorBoard 日志已生成在 `{run_logdir}`，可通过：")
    print(f"    tensorboard --logdir={os.path.dirname(run_logdir)}\n")

if __name__ == "__main__":
    main()
