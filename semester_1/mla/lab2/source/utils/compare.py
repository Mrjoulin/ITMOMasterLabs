from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from models import ParzenWindowKNN
from .metrics import get_metrics

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def compare_with_sklearn(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, best_k: int,
    X_prototypes: Optional[np.ndarray] = None, y_prototypes: Optional[np.ndarray] = None
):
    eval_models = {
        "Our KNN": ParzenWindowKNN(k=best_k).fit(X_train, y_train),
        "Sklearn KNN": KNeighborsClassifier(n_neighbors=best_k).fit(X_train, y_train)
    }
    if X_prototypes is not None and y_prototypes is not None:
        eval_models["Our KNN (Prototypes)"] = ParzenWindowKNN(k=1).fit(X_prototypes, y_prototypes)
        eval_models["Sklearn KNN (Prototypes)"] = KNeighborsClassifier(n_neighbors=1).fit(X_prototypes, y_prototypes)

    metrics_data = []
    predictions = {}

    for name, model in eval_models.items():
        model_predictions = model.predict(X_test)
        model_metrics = get_metrics(y_test, model_predictions)

        predictions[name] = model_predictions
        metrics_data.append([name, len(X_test), *model_metrics])

    # Print metrics compare df
    metrics_df = pd.DataFrame(
        data=metrics_data,
        columns=["Method", "Test size", "Accuracy", "Precision", "Recall", "F1"]
    )
    print("\nMetrics comparison:")
    print(metrics_df)

    # Comparison of predictions
    coincidence_matrix = []
    for name1, preds1 in predictions.items():
        coincidence_matrix.append([])
        for name2, preds2 in predictions.items():
            coincidence_acc = np.mean(preds1 == preds2)
            coincidence_matrix[-1].append(coincidence_acc)

    coincidence_df = pd.DataFrame(coincidence_matrix, index=list(predictions), columns=list(predictions))

    print(f'\nCoincidence of predictions:')
    print(coincidence_df)
