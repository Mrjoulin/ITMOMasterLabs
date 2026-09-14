from datetime import datetime
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Input

from .data_loader import load_config, get_engine, load_dataset

CONFIG_PATH = "config.yaml"


def build_cnn_model(window_size, n_features, filters, kernel_size, pool_size, dense_units, dropout, learning_rate):
    model = Sequential([
        Input(shape=(window_size, n_features)),
        Conv1D(filters, kernel_size, activation="relu", padding="same"),
        MaxPooling1D(pool_size),
        Conv1D(filters * 2, kernel_size, activation="relu", padding="same"),
        MaxPooling1D(pool_size),
        Flatten(),
        Dense(dense_units),
        Dropout(dropout),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    config = load_config(CONFIG_PATH)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    engine = get_engine(config)
    X, y = load_dataset(engine, config)

    if len(X) < 10:
        print("Not enough data.")
        return

    np.savez("data/features.npz", X=X, y=y)

    split_idx = int(len(X) * (1 - config["data"]["test_size"]))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Normalize data (fit on train only)
    n_samples_train, window_size, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, window_size, n_features)
    X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, window_size, n_features)

    with mlflow.start_run(run_name=str(datetime.now())) as run:
        params = config["model"]["params"]
        mlflow.log_params({
            "model_type": "cnn1d",
            "window_size": config["data"]["window_size"],
            "n_features": n_features,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            **params
        })
        mlflow.log_artifact(CONFIG_PATH)
        mlflow.log_artifact("data/features.npz")

        # Save and log scaler
        scaler_path = "data/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        model = build_cnn_model(
            window_size=config["data"]["window_size"],
            n_features=n_features,
            filters=params["filters"],
            kernel_size=params["kernel_size"],
            pool_size=params["pool_size"],
            dense_units=params["dense_units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"]
        )

        history = model.fit(
            X_train_scaled, y_train,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            validation_data=(X_test_scaled, y_test),
            verbose=0,
            callbacks=[mlflow.tensorflow.MlflowCallback(run)]
        )

        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Hypothesis 3 (CNN 1D) - Confusion Matrix")
        plt.savefig("images/confusion_matrix.png")
        mlflow.log_artifact("images/confusion_matrix.png")
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="train")
        plt.plot(history.history["val_loss"], label="val")
        plt.title("Loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="train")
        plt.plot(history.history["val_accuracy"], label="val")
        plt.title("Accuracy")
        plt.legend()
        plt.savefig("images/training_history.png")
        mlflow.log_artifact("images/training_history.png")
        plt.close()

        mlflow.keras.log_model(model, "cnn")
        print(f"Hypothesis 3 (CNN 1D) done. Metrics: {metrics}")


if __name__ == "__main__":
    main()
