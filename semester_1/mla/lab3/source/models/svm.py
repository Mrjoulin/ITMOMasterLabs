import numpy as np
from scipy.optimize import minimize

from .kernel import Kernel


class CustomSVM:
    def __init__(self, C=1.0, kernel='linear', gamma='scale', degree=3):
        self.C = C

        if self.C <= 0:
            raise ValueError("Параметр C должен быть положителен")

        if isinstance(kernel, Kernel) or callable(kernel):
            self.kernel = kernel
        elif isinstance(kernel, str):
            self.kernel = Kernel(kernel, gamma=gamma, degree=degree)
        else:
            raise ValueError("Ядро должно быть либо строкой (названием), либо функцией")

        # Support vector lambdas, vectors, vectors labels 
        self._lambdas = None
        self._vectors = None
        self._labels = None
        self._bias = 0
        # For linear kernel save exact weights
        self._linear_w = None

    @staticmethod
    def objective_function(lambdas, weighted_kernel):
        # Целевая функция для двойственной задачи
        objective = 0.5 * np.sum(
            lambdas[:, np.newaxis] * lambdas[np.newaxis, :] * weighted_kernel
        ) - np.sum(lambdas)
        jacobian = np.sum(lambdas[np.newaxis, :] * weighted_kernel, axis=1) - 1
        return objective, jacobian

    def fit(self, X, y, max_iter=1000, eps: float = 1e-6, tol: float = 1e-4):
        if self.kernel.gamma == 'scale':
            gamma = 1.0 / (X.shape[1] * X.var())
            self.kernel.set_gamma(gamma)

        # Precalculate matrix y_i * y_j * K(x_1, x_2)
        weighted_kernel = np.outer(y, y) * self.kernel(X, X)

        # Init lambdas and their bounds
        initial_lambdas = np.zeros(X.shape[0])
        cnts = np.unique_counts(y)
        for idx, val in enumerate(cnts.values):
            n_vals = len(cnts.values)
            others_prop = 1 - cnts.counts[idx] / X.shape[0]
            initial_lambdas[y == val] = n_vals * val * others_prop

        bounds = [(0, self.C) for _ in range(X.shape[0])]

        constraints = [{
            'type': 'eq',
            'fun': lambda _lamb, _y: np.dot(_lamb, _y),  # sum(lambdas * y) = 0
            'jac': lambda _lamb, _y: _y,
            "args": (y,)
        }]

        # Minimize target objective function
        result = minimize(
            CustomSVM.objective_function, initial_lambdas, args=(weighted_kernel,),
            jac=True, method='SLSQP', bounds=bounds, constraints=constraints, tol=tol,
            options={'maxiter': max_iter, 'disp': False}
        )
        lambdas = result.x

        # Find support vectors (lambdas > eps)
        sup_vec_indices = lambdas > eps
        self._lambdas = lambdas[sup_vec_indices]
        self._vectors = X[sup_vec_indices]
        self._labels = y[sup_vec_indices]

        if len(self._lambdas) > 0:
            if self.kernel == 'linear':
                self._linear_w = np.sum(
                    self._lambdas[:, None] * self._labels[:, None] * self._vectors,
                    axis=0
                )

            # Calculate bias using support vectors
            kernel_vals = self.kernel(self._vectors, self._vectors[:1])
            self._bias = np.mean(
                np.sum(self._lambdas * self._labels * kernel_vals[:, 0], axis=0) - self._labels
            )
        else:
            self._linear_w = np.zeros(X.shape[1])
            self._bias = 0

    def predict(self, X):
        if self.kernel == 'linear':
            predictions = np.dot(X, self._linear_w) - self._bias
        else:
            kernel_matrix = self.kernel(X, self._vectors)
            predictions = np.dot(kernel_matrix, self._lambdas * self._labels) - self._bias
        return np.sign(predictions)
