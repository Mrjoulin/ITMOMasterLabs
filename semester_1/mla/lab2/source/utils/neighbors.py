import numpy as np


def compute_distance_matrix(X):
    n_samples = X.shape[0]
    distances = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def find_k_neighbors(distances, k, exclude_self=True):
    n_samples = distances.shape[0]
    indices = np.zeros((n_samples, k), dtype=int)

    for i in range(n_samples):
        # Obtain the distances from the i-th object to all the others
        row_distances = distances[i].copy()

        if exclude_self:
            # Exclude the object itself (set distance to itself = inf)
            row_distances[i] = np.inf

        # Find the indices of the k nearest neighbors
        neighbor_indices = np.argpartition(row_distances, k)[:k]
        # Sort by distances
        sorted_neighbors = neighbor_indices[np.argsort(row_distances[neighbor_indices])]
        indices[i] = sorted_neighbors

    return indices


def find_neighbors_in_subset(distances, subset_indices, k, exclude_self=True):
    n_samples = distances.shape[0]
    n_subset = len(subset_indices)

    # Can't find k neighbors in subset with k elements or less
    if n_subset <= k:
        return np.array([]), np.array([])

    indices = np.zeros((n_samples, k), dtype=int)
    neighbor_distances = np.zeros((n_samples, min(k, n_subset)))

    for i in range(n_samples):
        # Distances to objects in a subset only
        subset_dists = distances[i, subset_indices]

        if exclude_self and i in subset_indices:
            subset_dists[subset_indices.index(i)] = np.inf

        # Find the indices of the k nearest neighbors
        neighbor_indices = np.argpartition(subset_dists, k)[:k]
        sorted_idx = neighbor_indices[np.argsort(subset_dists[neighbor_indices])]
        # Save original indexes and neighbors distances
        indices[i] = [subset_indices[idx] for idx in sorted_idx]
        neighbor_distances[i] = subset_dists[sorted_idx]

    return indices
