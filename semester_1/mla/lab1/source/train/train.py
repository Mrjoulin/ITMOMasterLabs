import time
import warnings
import numpy as np

from sklearn.linear_model import SGDClassifier

from model import LinearClassifier
from metrics import accuracy_score
from .consts import LEARNING_RATE, MOMENTUM_BETTA, L2_COEF, LOSS_LAMBDA


def train_and_eval_model(
    model: LinearClassifier, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 100, batch_size: int = 32, verbose: int = 0
):
    history, val_history = model.fit(
        X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, verbose=verbose
    )
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, (history, val_history)


def multi_start(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, loss: str = "log_loss",
    num_starts: int = 10, batch_method: str = "margin", epochs=100, batch_size=32, log_prefix: str = ""
):
    best_model = None
    best_model_acc = None
    best_hist = None
    for i in range(num_starts):
        print(f"{log_prefix}Train model {i + 1}/{num_starts}")
        model = LinearClassifier(
            weights_init_method="random",
            batch_method=batch_method,
            loss=loss,
            learning_rate=LEARNING_RATE,
            momentum_betta=MOMENTUM_BETTA,
            l2_coef=L2_COEF,
            loss_lambda=LOSS_LAMBDA
        )
        accuracy, hist = train_and_eval_model(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, batch_size=batch_size, verbose=0
        )
        if best_model_acc is None or best_model_acc < accuracy:
            best_model = model
            best_model_acc = accuracy
            best_hist = hist
            print(f"{log_prefix}Set best accuracy to {accuracy:.4f}")

    return best_model, best_model_acc, best_hist


def train_pipeline(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
    loss: str = "log_loss", num_starts: int = 10, epochs: int = 100, batch_size: int = 32
):
    # Multi-starts with random weights init

    print("Start Train pipeline with params:")
    print(f"-- Loss name: {loss}")
    print(f"-- Multi-start number of starts: {num_starts}")
    print(f"-- Train epochs: {epochs}")
    print(f"-- Batch size: {batch_size}")

    start_time = time.time()

    print("-- Start multi-start with margin-based batch method")
    best_model, best_model_acc, best_hist = multi_start(
        X_train, y_train, X_test, y_test,
        loss=loss, num_starts=num_starts, batch_method="margin",
        epochs=epochs, batch_size=batch_size, log_prefix="---- "
    )
    print(f"-- Complete multi-start with margin-based batch method with best accuracy {best_model_acc:.4f}")
    print("-- Start multi-start with random batch method")
    model, accuracy, hist = multi_start(
        X_train, y_train, X_test, y_test,
        loss=loss, num_starts=num_starts, batch_method="random",
        epochs=epochs, batch_size=batch_size, log_prefix="---- "
    )
    print(f"-- Complete multi-start with random batch method with best accuracy {accuracy:.4f}")

    if accuracy > best_model_acc:
        best_model, best_model_acc, best_hist = model, accuracy, hist
        print(f"-- Set new best accuracy to {accuracy:.4f}")

    # Train with weights init by correlation

    print("-- Start train with weights init by correlation and margin-based batch method")
    model = LinearClassifier(
        weights_init_method="correlation",
        batch_method="margin",
        loss=loss,
        learning_rate=LEARNING_RATE,
        momentum_betta=MOMENTUM_BETTA,
        l2_coef=L2_COEF,
        loss_lambda=LOSS_LAMBDA
    )
    accuracy, hist = train_and_eval_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    print(f"-- Complete train with weights init by correlation and "
          f"margin-based batch method with accuracy {accuracy:.4f}")

    if accuracy > best_model_acc:
        best_model, best_model_acc, best_hist = model, accuracy, hist
        print(f"-- Set new best accuracy to {accuracy:.4f}")

    print("-- Start train with weights init by correlation and random batch method")
    model = LinearClassifier(
        weights_init_method="correlation",
        batch_method="random",
        loss=loss,
        learning_rate=LEARNING_RATE,
        momentum_betta=MOMENTUM_BETTA,
        l2_coef=L2_COEF,
        loss_lambda=LOSS_LAMBDA
    )
    accuracy, hist = train_and_eval_model(
        model, X_train, y_train, X_test, y_test,
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    print(f"-- Complete train with weights init by correlation and random batch method with accuracy {accuracy:.4f}")

    if accuracy > best_model_acc:
        best_model, best_model_acc, best_hist = model, accuracy, hist
        print(f"-- Set new best accuracy to {accuracy:.4f}")

    print(f"End Train pipeline in {time.time() - start_time:.2f} sec, best accuracy: {best_model_acc:.4f}")

    return best_model, best_hist


def train_sklearn_models(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, epochs: int = 100
):
    print("Train SKLearn models")
    start_time = time.time()

    # Sklearn prints some RuntimeWarning-s, disable them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        print("-- Train LogisticRegression with SGD")
        logistic_clf = SGDClassifier(
            loss="log_loss", penalty="l2", max_iter=epochs,
            learning_rate='constant', eta0=LEARNING_RATE
        )
        logistic_clf.fit(X_train, y_train)
        logistic_score = logistic_clf.score(X_test, y_test)
        print(f"-- LogisticRegression accuracy: {logistic_score:.4f}")

        print("-- Train SVC (Support Vector Machine Classifier) with SGD")
        svc_clf = SGDClassifier(
            loss="hinge", penalty="l2", max_iter=epochs,
            learning_rate='constant', eta0=LEARNING_RATE
        )
        svc_clf.fit(X_train, y_train)
        svc_score = svc_clf.score(X_test, y_test)
        print(f"-- SVC accuracy: {svc_score:.4f}")

    best_accuracy = max(logistic_score, svc_score)

    print(f"End training SKLearn models in {time.time() - start_time:.2f} sec, best accuracy {best_accuracy:.4f}")

    return logistic_clf if logistic_score > svc_score else svc_clf
