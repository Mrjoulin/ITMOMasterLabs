import time
from datetime import datetime
from itertools import chain

import mlflow
import mlflow.sklearn
import structlog
import numpy as np
from prefect import task
from prefect.cache_policies import NO_CACHE
from sklearn.ensemble import RandomForestClassifier

from src.db.postgres import get_engine, load_sampled_price_time_series
from src.features.calc import aggregate_features
from src.utils.params_optimize import optuna_tune
from src.utils.metrics import eval_metrics
from src.utils.utils import DEFAULT_CONFIG_PATH, REQUIREMENTS_PATH, plot_confusion_matrix, plot_feature_importance


SPLIT_GROUP_SIZE = 20
logger = structlog.get_logger()


@task(name="Features collect", log_prints=True, cache_policy=NO_CACHE)
def features_collect(config):
    engine = get_engine()

    window = config["data"]["window_seconds"]
    event_duration = config["data"]["event_duration"]
    n_events = config["data"].get("n_events", None)
    target_threshold = config["data"]["target_threshold"]

    logger.info("Start collecting data")
    start_time = time.time()
    df = load_sampled_price_time_series(
        engine, window_seconds=window, event_duration_sec=event_duration, n_events=n_events
    )
    logger.info(f"Data collected in {time.time() - start_time:,.2f} sec")

    last_fix_ts = df["fix_ts"].iloc[-1]

    logger.info("Start aggregation data")
    start_time = time.time()
    df = aggregate_features(df, target_threshold=target_threshold)
    logger.info(f"Data aggregated in {time.time() - start_time:,.2f} sec")

    return df, last_fix_ts


def train_test_split(config, df):
    X, y, y_bin = df.drop(["interval_idx", "target", "bin_target"], axis=1), df["target"], df["bin_target"]

    # Take test percent from each group of 20 rows, leave last not full group as is (fresh data)
    n_groups = int(X.shape[0] / SPLIT_GROUP_SIZE)
    if n_groups == 0:
        logger.error("Not enough data.", n_rows=X.shape[0])
        raise RuntimeError("Not enough data")

    n_test_in_group = int(SPLIT_GROUP_SIZE * config["data"]["test_size"])
    test_idx = list(chain(*[
        np.random.choice(
            np.arange(i * SPLIT_GROUP_SIZE, (i + 1) * SPLIT_GROUP_SIZE),
            size=n_test_in_group, replace=False
        )
        for i in range(n_groups)
    ]))

    # Split to train/test
    X_train, X_test = X.drop(test_idx), X.iloc[test_idx]
    y_train, y_test = y.drop(test_idx), y.iloc[test_idx]
    y_bin_train, y_bin_test = y_bin.drop(test_idx), y_bin.iloc[test_idx]
    dataset = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_bin_train": y_bin_train,
        "y_bin_test": y_bin_test
    }
    return dataset


@task(name="Train model", log_prints=True, cache_policy=NO_CACHE)
def run_model_train(config, dataset):
    if len(dataset["X_train"]) < SPLIT_GROUP_SIZE:
        logger.error("Not enough data.", n_rows=len(dataset["X_train"]))
        raise RuntimeError("Not enough data")

    logger.info("Train targets distribution:", targets=dataset["y_train"].value_counts().to_dict())
    logger.info("Test targets distribution:", targets=dataset["y_test"].value_counts().to_dict())

    use_optuna = config["model"].get("use_optuna", False)
    if use_optuna:
        try:
            model_params = optuna_tune(config, dataset, base_estimator=RandomForestClassifier)
            config["model"]["params"] = model_params
        except Exception as e:
            logger.error("Unable to tune params with optuna", error=e)
            config["model"]["use_optuna"] = False

    model = RandomForestClassifier(**config["model"]["params"])
    model.fit(dataset["X_train"], dataset["y_train"])
    metrics = eval_metrics(model, dataset["X_test"], dataset["y_test"], dataset["y_bin_test"])

    return model, metrics


@task(name="Log to MLFlow", log_prints=True, cache_policy=NO_CACHE)
def log_model_mlflow(config, model, dataset, last_fix_ts):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    train_features = list(dataset["X_train"].columns)

    with mlflow.start_run(run_name=str(datetime.now())):
        mlflow.log_params({
            "model_type": config["model"]["type"],
            "optuna_used": config["model"]["use_optuna"],
            "n_features": dataset["X_train"].shape[1],
            "train_samples": len(dataset["X_train"]),
            "test_samples": len(dataset["X_test"]),
            "train_features": train_features,
            "last_fix_ts": int(last_fix_ts),
            **config["data"],
            **config["model"]["params"]
        })
        mlflow.log_artifact(DEFAULT_CONFIG_PATH)
        mlflow.sklearn.log_model(
            model, name=config["model"]["type"],
            registered_model_name=config["mlflow"]["registered_model"],
            pip_requirements=REQUIREMENTS_PATH
        )
        logger.info("Model fitted and logged to MLflow")

        # Calc metrics
        metrics = eval_metrics(model, dataset["X_train"], dataset["y_train"], dataset["y_bin_train"], prefix="train")
        metrics.update(eval_metrics(model, dataset["X_test"], dataset["y_test"], dataset["y_bin_test"], prefix="test"))
        mlflow.log_metrics(metrics)

        # Plotting
        y_pred = model.predict(dataset["X_test"])
        images_artefacts = [
            plot_confusion_matrix(dataset["y_test"], y_pred),
            plot_feature_importance(model, train_features)
        ]
        for img_path in images_artefacts:
            if img_path:
                mlflow.log_artifact(img_path)

        logger.info(f"Train Done", metrics=metrics)

    return metrics
