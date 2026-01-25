import numpy as np

from .compactness import CompactnessProfile
from utils import find_neighbors_in_subset


class PrototypeSelector:
    def __init__(self, k=3):
        self.k = k
        self.profile_calc = None
        self.X = None
        self.y = None
        self.L = None
        self.distances = None

        self.omega_indices = None
        self.history = {}

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.L = len(y)
        self.profile_calc = CompactnessProfile()
        self.profile_calc.fit(X, y)
        self.distances = self.profile_calc.distances

        # Initialization: one object from each class
        self._initialize_prototypes()

        # Computing the initial CCV
        current_ccv = self.profile_calc.compute_ccv(self.omega_indices, self.k)

        print("\nStart of prototype selection:")
        print(f"Initialization: |Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}")

        iteration = 0
        ccv_history = [current_ccv]
        omega_sizes = [len(self.omega_indices)]

        # Greedy prototype addition strategy
        while len(self.omega_indices) < self.L:
            iteration += 1
            best_candidate = None
            best_ccv = current_ccv

            # Looking for candidates to add
            candidates = self._find_promising_candidates()

            for candidate in candidates:
                new_ccv = self.profile_calc.compute_ccv(self.omega_indices + [candidate], self.k)

                if new_ccv < best_ccv:
                    best_ccv = new_ccv
                    best_candidate = candidate

            if best_candidate is not None and best_ccv < current_ccv:
                self.omega_indices.append(best_candidate)
                current_ccv = best_ccv
                ccv_history.append(best_ccv)
                omega_sizes.append(len(self.omega_indices))

                print(
                    f"Iteration {iteration}: added object {best_candidate}, "
                    f"|Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}"
                )
            else:
                print(f"Stopping at iteration {iteration}: no improvements found")
                print(f"Final |Ω| = {len(self.omega_indices)}, CCV = {current_ccv:.4f}")
                break

        self.history = {
            "ccv": ccv_history,
            "omega_sizes": omega_sizes
        }

        return self

    def _initialize_prototypes(self):
        unique_classes = np.unique(self.y)
        self.omega_indices = []

        for cls in unique_classes:
            class_indices = np.where(self.y == cls)[0]
            if len(class_indices) > 0:
                # Select the object closest to the class centroid
                centroid = np.mean(self.X[class_indices], axis=0)
                distances_to_centroid = [np.linalg.norm(self.X[i] - centroid) for i in class_indices]
                best_idx = class_indices[np.argmin(distances_to_centroid)]
                self.omega_indices.append(best_idx)

        return self.omega_indices

    def _find_promising_candidates(self, n_candidates=20):
        candidates = np.array([i for i in range(self.L) if i not in self.omega_indices])

        if len(candidates) <= n_candidates:
            return candidates

        # Evaluate the "improvement" of each candidate
        candidate_scores = np.zeros_like(candidates)

        for i, candidate in enumerate(candidates):
            # Quick estimation: check how many objects will be classified better
            improvement = self._estimate_improvement(candidate)
            # Save it with a negative sign for easy sorting
            candidate_scores[i] = -improvement

        # Choose the best in terms of "improvement"
        candidates_order_idx = np.argpartition(candidate_scores, n_candidates)[:n_candidates]
        sorted_candidates_idx = candidates_order_idx[np.argsort(candidate_scores[candidates_order_idx])]
        return candidates[sorted_candidates_idx]

    def _estimate_improvement(self, candidate):
        improvement = 0

        # Find neighbors in the current subset
        current_neighbors = find_neighbors_in_subset(self.distances, self.omega_indices, 1, exclude_self=True)

        # Find for which objects the candidate will become the nearest neighbor
        for i in range(self.L):
            if i == candidate:
                continue

            dist_to_candidate = self.distances[i, candidate]
            candidate_class = self.y[candidate]

            # Check if the candidate will be closer than current neighbors
            current_neighbor_idx = current_neighbors[i, 0]
            current_neighbor_dist = self.distances[i, current_neighbor_idx]
            current_neighbor_class = self.y[current_neighbor_idx]
            if dist_to_candidate < current_neighbor_dist and candidate_class != current_neighbor_class:
                # If the candidate is of the right class, this is an improvement
                if candidate_class == self.y[i]:
                    improvement += 1
                # If the current neighbor was of the right class and the candidate was of the wrong class,
                # then this is a deterioration
                elif current_neighbor_class == self.y[i]:
                    improvement -= 1

        return improvement

    def get_prototypes(self):
        return self.X[self.omega_indices], self.y[self.omega_indices]
