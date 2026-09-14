import pandas as pd
from .metrics import evaluate_model, eval_model, get_metrics

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def train_eval_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    print("Evaluation:")
    cm = evaluate_model(model, X_test, y_test, log_prefix="  ")
    return cm


def compare_with_sklearn(models, X_test, y_test):
    """
    Compare our implementation of Logistic Regression with sklearn's implementation.
    """
    print("=" * 60)
    print("COMPARING LOGISTIC REGRESSION IMPLEMENTATIONS")
    print("=" * 60)

    metrics = []
    scores = {}
    for model_name, model in models.items():
        model_metrics = [model_name, X_test.shape[0]]
        y_pred, y_scores = eval_model(model, X_test)
        model_metrics += list(get_metrics(y_test, y_pred, y_scores))
        metrics.append(model_metrics)
        scores[model_name] = y_scores

    metrics_df = pd.DataFrame(
        metrics,
        columns=["Method", "Test size", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
    )
    print("\nMetrics comparison:")
    print(metrics_df)

    return scores
