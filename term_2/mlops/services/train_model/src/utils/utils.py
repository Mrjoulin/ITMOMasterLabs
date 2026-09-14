import os
import yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

DEFAULT_CONFIG_PATH = "configs/config.yaml"
IMAGES_SAVE_PATH = "images/"


def load_config(config_path=None):
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    pg = config["postgres"]
    pg["user"] = os.path.expandvars(pg["user"])
    pg["password"] = os.path.expandvars(pg["password"])
    pg["dbname"] = os.path.expandvars(pg["dbname"])

    return config


def check_pos_int(val: int) -> bool:
    return val is not None and isinstance(val, int) and str(val).isdigit() and val >= 0


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
