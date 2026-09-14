import structlog
import numpy as np
from prefect import flow
from src.train import features_collect, train_test_split, run_model_train, log_model_mlflow
from src.inference import make_model_request
from src.utils.utils import load_config, get_mlflow_last_run_params
from src.utils.metrics import calc_metrics

logger = structlog.get_logger()


def compare_metrics(metrics_new: dict, metrics_old: dict):
    common_keys = set(metrics_new.keys()) & set(metrics_old.keys())
    metrics_compare = {True: [], False: []}
    for key in common_keys:
        if key != "mse":
            metrics_compare[metrics_new[key] > metrics_old[key]].append(key)
        else:
            metrics_compare[metrics_new[key] < metrics_old[key]].append(key)
    return metrics_compare[True], metrics_compare[False]


@flow(name="model-train", log_prints=True)
def model_train_flow():
    try:
        config = load_config()
        df, last_fix_ts = features_collect(config)

        last_run_params = get_mlflow_last_run_params(config["mlflow"]["experiment_name"])
        if last_run_params and last_run_params.get("last_fix_ts") is not None:
            if last_run_params["last_fix_ts"] == last_fix_ts:
                logger.warning(f"Same data received as in last run with last_fix_ts = {last_fix_ts}!")
                logger.info("Stop training pipeline")
                return
        elif not last_run_params:
            logger.warning("Previous run params not found in mlflow!")
        else:
            logger.warning("last_fix_ts not found in previous run params in mlflow!")

        dataset = train_test_split(config, df)

        # Eval previous model
        try:
            last_model_preds = make_model_request(config, dataset["X_test"])
            last_model_preds = np.array(last_model_preds)
            last_model_metrics = calc_metrics(dataset["y_test"], dataset["y_bin_test"], last_model_preds)
        except Exception as e:
            logger.warning("Unable to eval previous model", error=e)
            last_model_metrics = None

        # Train new model
        model, metrics = run_model_train(config, dataset)

        # QUALITY GATE: Compare metrics with prev model
        if last_model_metrics is not None:
            n_tries, try_number = 2, 0
            while try_number < n_tries:
                try_number += 1
                better_metrics, worse_metrics = compare_metrics(metrics, last_model_metrics)
                if worse_metrics:
                    logger.warning(f"Trained model showed worse metrics then previous", worse_metrics=worse_metrics)

                if len(better_metrics) < len(worse_metrics):
                    logger.warning(
                        f"Trained model showed too much worse metrics then previous",
                        cnt_better=len(better_metrics), cnt_worse=len(worse_metrics)
                    )
                    if not config["model"]["use_optuna"]:
                        logger.info("Tune model with optune")

                        config["model"]["use_optuna"] = True
                        model, metrics = run_model_train(config, dataset)
                    else:
                        logger.info("Early stop training pipeline, worse metrics")
                        return
                else:
                    logger.info("Model showed better performance on at least half of metrics")
                    break

        log_model_mlflow(config, model, dataset, last_fix_ts)
    except RuntimeError as e:
        logger.error(f"Stop train pipeline: {e}")
        raise e


if __name__ == "__main__":
    model_train_flow()
