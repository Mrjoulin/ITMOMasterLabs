import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import load_config, get_engine, load_events, load_time_series
from .feature_engineer import extract_features

CONFIG_PATH = "config.yaml"


def main():
    config = load_config(CONFIG_PATH)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    engine = get_engine(config)
    events = load_events(engine)
    window = config["data"]["window_seconds"]

    features = []
    for _, row in events.iterrows():
        ts = load_time_series(engine, row["event_start_ts"], window)
        if ts.empty:
            continue
        feats = extract_features(ts, row["start_price"])
        if feats:
            feats["target"] = row["target"]
            features.append(feats)

    data = pd.DataFrame(features)
    if len(data) < 10:
        print("Not enough data.")
        return

    data.to_csv("data/features.csv")

    X = data.drop("target", axis=1)
    y = data["target"]
    split_idx = int(len(data) * (1 - config["data"]["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    with mlflow.start_run(run_name=str(datetime.now())):
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "window_seconds": window,
            "n_features": X.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_features": list(X_train.columns),
            "events_first_ts": events["event_start_ts"].min(),
            "events_last_ts": events["event_start_ts"].max(),
            **config["model"]["params"]
        })
        mlflow.log_artifact(CONFIG_PATH)
        mlflow.log_artifact("data/features.csv")

        model = RandomForestClassifier(**config["model"]["params"])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Hypothesis 1 (Technical RF) - Confusion Matrix")
        plt.savefig("images/confusion_matrix.png")
        mlflow.log_artifact("images/confusion_matrix.png")
        plt.close()

        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame(
                {"feature": X.columns, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)
            plt.figure(figsize=(12, 6))
            sns.barplot(x="importance", y="feature", data=imp_df.head(20))
            plt.title("Top 20 Feature Importances - Hypothesis 1")
            plt.savefig("images/feature_importances.png")
            mlflow.log_artifact("images/feature_importances.png")
            plt.close()

        mlflow.sklearn.log_model(model, config["model"]["type"])
        print(f"Hypothesis 1 done. Metrics: {metrics}")
