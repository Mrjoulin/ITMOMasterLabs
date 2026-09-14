import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, confusion_matrix


def calc_metrics(y, y_bin, y_pred, y_proba=None, classes=None, prefix: str = None):

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "mse": np.mean((y_pred - y) ** 2)
    }
    if y_proba is not None:
        if classes is None:
            classes = [-1, 0, 1]
        metrics["roc_auc"] = roc_auc_score(y_pred, y_proba, labels=classes, multi_class='ovr')

    cm = confusion_matrix(y, y_pred)
    y_pred_buy = y_pred[y_pred != 0]
    y_bin_buy = y_bin[y_pred != 0]

    metrics.update({
        "precision_UP": precision_score(y_bin_buy, y_pred_buy),
        "recall_UP": cm[2, 2] / cm[2].sum(),
        "precision_DOWN": precision_score(y_bin_buy, y_pred_buy, pos_label=-1),
        "recall_DOWN": cm[0, 0] / cm[0].sum()
    })

    metrics.update({
        "f1_UP": (2 * metrics["precision_UP"] * metrics["recall_UP"]) /
                          (metrics["precision_UP"] + metrics["recall_UP"]),
        "f1_DOWN": (2 * metrics["precision_DOWN"] * metrics["recall_DOWN"]) /
                            (metrics["precision_DOWN"] + metrics["recall_DOWN"])
    })
    metrics["meta_f1"] = (
            (2 * metrics["f1_UP"] * metrics["f1_DOWN"]) /
            (metrics["f1_UP"] + metrics["f1_DOWN"])
    )
    metrics["avg_f1"] = (metrics["f1_UP"] + metrics["f1_DOWN"]) / 2

    if prefix is not None:
        prefix = f'{prefix.lstrip("_")}_'
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    return metrics


def eval_metrics(model, X, y, y_bin, prefix: str = None):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    metrics = calc_metrics(y, y_bin, y_pred, y_proba=y_proba, classes=model.classes_, prefix=prefix)
    return metrics
