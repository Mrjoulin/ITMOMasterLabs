import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import structlog
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix

from src.utils.postgres import get_engine, load_sampled_price_time_series
from src.utils.utils import DEFAULT_CONFIG_PATH, load_config, plot_confusion_matrix, plot_feature_importance
from src.features.calc import aggregate_features

logger = structlog.get_logger()


def optuna_tune(config, X_train, X_test, y_train, y_test, y_bin_test):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARN)

    static_params = config["model"]["params"].copy()
    optuna_params = config["model"]["optuna"]["params"]
    tune_params = config["model"]["optuna"]['tune'].copy()

    for param in tune_params:
        if param in static_params:
            static_params.pop(param)

    def train_eval_model(params):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        # Calc metrics
        metrics = calc_metrics(model, X_test, y_test, y_bin_test)
        return metrics

    def objective(trial: optuna.Trial):
        test_params = static_params.copy()
        for param, param_dict in tune_params.items():
            if param_dict["type"] == "int":
                test_params[param] = trial.suggest_int(param, *param_dict["range"])
            elif param_dict["type"] == "float":
                test_params[param] = trial.suggest_float(param, *param_dict["range"])

        metrics = train_eval_model(test_params)
        return metrics[optuna_params["metric"]]

    def logging_callback(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            logger.info(
                f"Trial {frozen_trial.number} finished with best value: {frozen_trial.value} "
                f"and parameters: {frozen_trial.params}. "
            )

    logger.info("Start parameters tune with optuna")
    study = optuna.create_study(direction=optuna_params["optimize_direction"])
    study.optimize(objective, callbacks=[logging_callback], n_trials=optuna_params["n_trials"])
    logger.info(f"Best params is {study.best_params} with value {study.best_value}")

    best_params = static_params.copy()
    best_params.update(study.best_params)
    return best_params


def calc_metrics(model, X, y, y_bin, prefix: str = None):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    prefix = "" if prefix is None else f'{prefix.lstrip("_")}_'

    metrics = {
        f"{prefix}accuracy": accuracy_score(y, y_pred),
        f"{prefix}roc_auc": roc_auc_score(y_pred, y_proba, labels=model.classes_, multi_class='ovr'),
        f"{prefix}mse": np.mean((y_pred - y)**2)
    }

    cm = confusion_matrix(y, y_pred)
    y_pred_buy = y_pred[y_pred != 0]
    y_bin_buy = y_bin[y_pred != 0]

    metrics.update({
        f"{prefix}precision_UP": precision_score(y_bin_buy, y_pred_buy),
        f"{prefix}recall_UP": cm[2, 2] / cm[2].sum(),
        f"{prefix}precision_DOWN": precision_score(y_bin_buy, y_pred_buy, pos_label=-1),
        f"{prefix}recall_DOWN": cm[0, 0] / cm[0].sum()
    })

    metrics.update({
        f"{prefix}f1_UP": (2 * metrics[f"{prefix}precision_UP"] * metrics[f"{prefix}recall_UP"]) /
                          (metrics[f"{prefix}precision_UP"] + metrics[f"{prefix}recall_UP"]),
        f"{prefix}f1_DOWN": (2 * metrics[f"{prefix}precision_DOWN"] * metrics[f"{prefix}recall_DOWN"]) /
                            (metrics[f"{prefix}precision_DOWN"] + metrics[f"{prefix}recall_DOWN"])
    })
    metrics[f"{prefix}meta_f1"] = (
        (2 * metrics[f"{prefix}f1_UP"] * metrics[f"{prefix}f1_DOWN"]) /
        (metrics[f"{prefix}f1_UP"] + metrics[f"{prefix}f1_DOWN"])
    )

    return metrics


def main():
    config = load_config(DEFAULT_CONFIG_PATH)
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    engine = get_engine(config)
    window = config["data"]["window_seconds"]
    event_duration = config["data"]["event_duration"]
    target_threshold = config["data"]["target_threshold"]

    logger.info("Start collecting data")
    start_time = time.time()
    df = load_sampled_price_time_series(
        engine, window_seconds=window, event_duration_sec=event_duration
    )
    logger.info(f"Data collected in {time.time() - start_time:,.2f} sec")

    logger.info("Start aggregation data")
    start_time = time.time()
    df = aggregate_features(df, target_threshold=target_threshold)
    logger.info(f"Data aggregated in {time.time() - start_time:,.2f} sec")

    if len(df) < 10:
        logger.error("Not enough data.", n_rows=len(df))
        return

    X, y, y_bin = df.drop(["interval_idx", "target", "bin_target"], axis=1), df["target"], df["bin_target"]
    split_idx = int(len(df) * (1 - config["data"]["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    y_bin_train, y_bin_test = y_bin.iloc[:split_idx], y_bin.iloc[split_idx:]

    logger.info("Overall targets distribution:", targets=df["target"].value_counts().to_dict())
    logger.info("Train targets distribution:", targets=y_train.value_counts().to_dict())
    logger.info("Test targets distribution:", targets=y_test.value_counts().to_dict())

    if config["model"].get("use_optuna", False):
        model_params = optuna_tune(config, X_train, X_test, y_train, y_test, y_bin_test)
    else:
        model_params = config["model"]["params"]

    mlflow.sklearn.autolog(
        log_datasets=config["mlflow"]["log_datasets"],
        registered_model_name=config["mlflow"]["registered_model"]
    )

    with mlflow.start_run(run_name=str(datetime.now())):
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "window_seconds": window,
            "n_features": X.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_features": list(X_train.columns),
            **model_params
        })
        mlflow.log_artifact(DEFAULT_CONFIG_PATH)

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Calc metrics
        metrics = calc_metrics(model, X_train, y_train, y_bin_train, prefix="train")
        metrics.update(calc_metrics(model, X_test, y_test, y_bin_test, prefix="test"))
        mlflow.log_metrics(metrics)

        # Plotting
        y_pred = model.predict(X_test)
        images_artefacts = [
            plot_confusion_matrix(y_test, y_pred),
            plot_feature_importance(model, X.columns)
        ]
        for img_path in images_artefacts:
            if img_path:
                mlflow.log_artifact(img_path)

        logger.info(f"Train Done", metrics=metrics)

    mlflow.search_model_versions()

if __name__ == '__main__':
    main()
