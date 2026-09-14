import time
import structlog
from prefect import task
from prefect.cache_policies import NO_CACHE
from sklearn.ensemble import RandomForestClassifier

from .metrics import eval_metrics

logger = structlog.get_logger()
OPTUNE_MAX_TIME = 180  # 3 mins


@task(name="Hyperparams optimize", log_prints=True, cache_policy=NO_CACHE)
def optuna_tune(config, dataset, base_estimator=RandomForestClassifier):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARN)
    start_time = time.time()

    static_params = config["model"]["params"].copy()
    optuna_params = config["model"]["optuna"]["params"]
    tune_params = config["model"]["optuna"]['tune'].copy()

    for param in tune_params:
        if param in static_params:
            static_params.pop(param)

    def train_eval_model(params):
        model = base_estimator(**params)
        model.fit(dataset["X_train"], dataset["y_train"])
        # Calc metrics
        metrics = eval_metrics(model, dataset["X_test"], dataset["y_test"], dataset["y_bin_test"])
        return metrics

    def objective(trial: optuna.Trial):
        if time.time() - start_time >= OPTUNE_MAX_TIME:
            raise RuntimeError("Stopped by time limit")

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
    try:
        study.optimize(objective, callbacks=[logging_callback], n_trials=optuna_params["n_trials"])
    except RuntimeError as e:
        logger.warning(f"Optuna stopped: {e}")
    logger.info(f"Best params is {study.best_params} with value {study.best_value}")

    best_params = static_params.copy()
    best_params.update(study.best_params)
    return best_params
