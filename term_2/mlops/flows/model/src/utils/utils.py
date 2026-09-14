import os
import yaml
import time
import mlflow
import asyncio
import threading
from datetime import datetime
from typing import Union, Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DEFAULT_CONFIG_PATH = "model/configs/config.yaml"
REQUIREMENTS_PATH = "model/configs/requirements.txt"
IMAGES_SAVE_PATH = "model/images/"


def load_config(config_path=None):
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def check_pos_int(val: int) -> bool:
    return val is not None and isinstance(val, int) and str(val).isdigit() and val >= 0


def get_cur_ts(precision: Optional[str] = None) -> Union[int, float]:
    if precision is None:
        return time.time()
    elif precision.startswith("sec"):
        return round(time.time())
    elif precision.startswith("milli"):
        return round(time.time() * 1000)
    else:
        return time.time()


def ts_to_dt(ts: int, to_str: bool = True) -> Union[datetime, str]:
    if ts > 1e12:
        ts = ts / 1000
    if to_str:
        return str(datetime.fromtimestamp(ts))
    else:
        return datetime.fromtimestamp(ts)


def run_in_thread(coroutine):
    threading.Thread(
        target=asyncio.run, args=(coroutine,), daemon=True
    ).start()


def _save_and_close(img_name: str):
    save_path = os.path.join(IMAGES_SAVE_PATH, img_name)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_matrix(y_test, y_pred):
    classes = [-1, 0, 1]

    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, index=classes, columns=classes)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title("Test Confusion Matrix")

    return _save_and_close("confusion_matrix.png")


def plot_feature_importance(model, columns):
    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame(
            {"feature": columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 10))
        sns.barplot(x="importance", y="feature", data=imp_df.head(40))
        plt.title("Top 40 Feature Importances")
        return _save_and_close("feature_importances.png")
    return None


def get_mlflow_last_run_params(experiment_name: str) -> dict:
    # 1. Search for runs, ordered by start time (newest first), and fetch only the top 1
    runs_df = mlflow.search_runs(
        experiment_names=[experiment_name],
        max_results=1,
        order_by=["start_time DESC"],
        output_format="list"
    )
    if len(runs_df) == 0:
        return {}
    run = runs_df[0]
    return run.data.params
