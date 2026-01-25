import time
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.svm import SVC

from models import CustomSVM
from .metrics import get_metrics

pd.set_option('display.max_columns', 10)
pd.set_option('display.expand_frame_repr', False)


def _parallel_train(model, kernel, X_train, y_train, return_dict):
    start_time = time.time()
    model.fit(X_train, y_train)
    total_time = time.time() - start_time

    print(f"- Done {kernel} in {total_time:.2f}")
    return_dict[kernel] = {
        "model": model,
        "total_time": total_time
    }


def _parallel_train_ours(eval_models, X_train, y_train):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for kernel, model in eval_models["Our SVM"].items():
        p = multiprocessing.Process(target=_parallel_train, args=(model, kernel, X_train, y_train, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    times = {}
    for kernel in return_dict:
        eval_models["Our SVM"][kernel] = return_dict[kernel]["model"]
        times[kernel] = return_dict[kernel]["total_time"]

    return times


def compare_with_sklearn(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, C: float = 1.0
):
    kernels = ['linear', 'poly', 'rbf']
    eval_models = {
        "Our SVM": {},
        "Sklearn SVM": {}
    }
    extra_params = dict(C=C, gamma='scale', degree=3)

    for kernel in kernels:
        eval_models["Our SVM"][kernel] = CustomSVM(kernel=kernel, **extra_params)
        eval_models["Sklearn SVM"][kernel] = SVC(kernel=kernel, **extra_params)

    print(f"Start training {len(eval_models['Our SVM'])} our and {len(eval_models['Sklearn SVM'])} sklearn SVMs")

    our_times = _parallel_train_ours(eval_models, X_train, y_train)

    metrics_data = []
    predictions = {}

    for name in eval_models:
        predictions[name] = {}
        for kernel in kernels:
            model = eval_models[name][kernel]

            if name != "Our SVM":
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = round(time.time() - start_time, 2)
            else:
                train_time = round(our_times.get(kernel), 2)

            model_predictions = model.predict(X_test)
            model_metrics = get_metrics(y_test, model_predictions)

            predictions[name][kernel] = model_predictions
            metrics_data.append([name, kernel, len(X_test), train_time, *model_metrics])

    # Print metrics compare df
    metrics_df = pd.DataFrame(
        data=metrics_data,
        columns=["Method", "Kernel", "Test size", "Train Time (sec)", "Accuracy", "Precision", "Recall", "F1"]
    )
    print("\nMetrics comparison:")
    print(metrics_df)

    return eval_models, predictions
