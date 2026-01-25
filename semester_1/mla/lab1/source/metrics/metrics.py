import numpy as np
import pandas as pd
import warnings


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == -1) & (y_pred == -1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return pd.DataFrame(
        [[tp, fp], [fn, tn]],
        columns=pd.MultiIndex.from_tuples([("Actual", "Positive"), ("Actual", "Negative")]),
        index=pd.MultiIndex.from_tuples([("Predict", "Positive"), ("Predict", "Negative")])
    )


def precision_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def recall_score(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, log_prefix: str = ""):
    # Sklearn models prints some RuntimeWarning-s, disable them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = model.predict(X_test)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print(f"{log_prefix}Accuracy: {accuracy:.4f}")
    print(f"{log_prefix}Precision: {precision:.4f}")
    print(f"{log_prefix}Recall: {recall:.4f}")
    print(f"{log_prefix}F1-Score: {f1:.4f}")

    return confusion_matrix(y_test, y_pred)
