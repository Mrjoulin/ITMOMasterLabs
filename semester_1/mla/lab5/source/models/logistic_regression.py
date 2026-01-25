import numpy as np


class LogisticRegression:
    def __init__(self, solver='nr', max_iter=200, tol=1e-3, learning_rate=0.01):
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.weights = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Add intercept term
        X = np.insert(X, 0, 1, axis=1)
        # Zeroth approximation
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

        if self.solver == 'nr':
            self._fit_newton_raphson(X, y)
        elif self.solver == 'irls':
            self._fit_irls(X, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

    def _fit_newton_raphson(self, X, y):
        prev_sigm = np.zeros_like(y)

        for i in range(self.max_iter):
            sigm = LogisticRegression._sigmoid(X @ self.weights * y)

            gradient = -(X.T @ (y / sigm))
            D = np.diag(sigm * (1 - sigm))
            hessian = X.T @ D @ X

            # Add regularization to prevent singular matrix
            hessian += 1e-6 * np.eye(hessian.shape[0])

            self.weights -= self.learning_rate * (np.linalg.inv(hessian) @ gradient)

            upd_norm = np.linalg.norm(sigm - prev_sigm)
            if upd_norm < self.tol:
                print(f"Converged at iteration {i + 1}")
                break
            else:
                prev_sigm = sigm
                print(f"-- NR iter {i + 1}: {upd_norm}")

    def _fit_irls(self, X, y):
        prev_sigm = np.zeros_like(y)

        for i in range(self.max_iter):
            sigm = LogisticRegression._sigmoid(X @ self.weights * y)
            gamma = np.sqrt(sigm * (1 - sigm))

            X = np.diag(gamma) @ X
            y = y * (gamma / sigm)

            self.weights += self.learning_rate * (np.linalg.inv(X.T @ X) @ X.T @ y)

            upd_norm = np.linalg.norm(sigm - prev_sigm)
            if upd_norm < self.tol:
                print(f"Converged at iteration {i + 1}")
                break
            else:
                prev_sigm = sigm
                print(f"-- IRLS iter {i + 1}: {upd_norm}")

    def predict_proba(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self._sigmoid(X @ self.weights)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, -1)
