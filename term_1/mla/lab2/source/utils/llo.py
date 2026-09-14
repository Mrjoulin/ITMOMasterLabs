import numpy as np
from models import ParzenWindowKNN


def leave_one_out(size: int):
    all_indices = np.arange(size)
    mask = np.ma.make_mask(all_indices)
    for i in all_indices:
        mask[i] = False
        yield all_indices[mask], np.array([i])
        mask[i] = True


def calculate_loo_accuracy(model: ParzenWindowKNN, X: np.ndarray, y: np.ndarray):
    n_samples = len(X)
    correct_predictions = 0

    for train_idx, test_idx in leave_one_out(X.shape[0]):
        model.fit(X[train_idx], y[train_idx])
        prediction = model.predict(X[test_idx])
        correct_predictions += (prediction == y[test_idx]).sum()
    return correct_predictions / n_samples


def find_best_k_loo(X: np.ndarray, y: np.ndarray, k_start: int = 1, k_end: int = 20, plot_graph: bool = True):
    best_k = None
    best_accuracy = 0
    accuracies = []

    print("\nSelecting the optimal k using the LOO method:")
    print("-" * 40)
    k_range = np.arange(int(k_start), int(k_end + 1))

    for k in k_range:
        knn = ParzenWindowKNN(k=k)
        accuracy = calculate_loo_accuracy(knn, X, y)
        accuracies.append(accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

        print(f"k = {k:2d}, Accuracy LOO: {accuracy:.4f}")

    print("-" * 40)
    print(f"Optimal k: {best_k} with accuracy {best_accuracy:.4f}")

    if plot_graph:
        from .plotting import plot_llo_graphs
        plot_llo_graphs(k_range, accuracies)

    return best_k

