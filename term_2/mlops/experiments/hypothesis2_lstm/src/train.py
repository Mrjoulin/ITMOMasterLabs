from datetime import datetime
import mlflow
import mlflow.keras
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input


from .data_loader import load_config, get_engine, load_dataset

CONFIG_PATH = "config.yaml"


def build_lstm_model(window_size, n_features, lstm_units, dropout, dense_units, learning_rate):
    model = Sequential([
        Input(shape=(window_size, n_features)),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
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

    # Reshape to 2D for scaling
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled_2d = scaler.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler.transform(X_test_2d)

    # Reshape back to 3D
    X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, window_size, n_features)
    X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, window_size, n_features)

    with mlflow.start_run(run_name=str(datetime.now())) as run:
        params = config["model"]["params"]
        mlflow.log_params({
            "model_type": "lstm",
            "window_size": config["data"]["window_size"],
            "n_features": n_features,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            **params
        })
        mlflow.log_artifact(CONFIG_PATH)
        mlflow.log_artifact("data/features.npz")

        # Save and log scaler
        import joblib
        scaler_path = "data/scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        model = build_lstm_model(
            window_size=config["data"]["window_size"],
            n_features=n_features,
            lstm_units=params["lstm_units"],
            dropout=params["dropout"],
            dense_units=params["dense_units"],
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
        plt.title("Hypothesis 2 (LSTM) - Confusion Matrix")
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

        mlflow.keras.log_model(model, "lstm")
        print(f"Hypothesis 2 (LSTM) done. Metrics: {metrics}")


if __name__ == "__main__":
    main()
