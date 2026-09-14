from typing import Optional
import numpy as np


class CustomPCA:
    """Реализация PCA через сингулярное разложение"""

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components

        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None

    def fit(self, X: np.ndarray):
        """
        Обучение PCA на данных X

        Параметры:
        X: array-like, shape (n_samples, n_features)
        """

        if X.shape[0] < X.shape[1]:
            raise ValueError(f"Invalid shape for X: {X.shape}! First dim should be >= second dim")
        if self.n_components is not None and self.n_components > X.shape[1]:
            raise ValueError(f"Invalid n_components = {self.n_components}! Should be <= second dim of X")

        # Центрирование данных
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Сингулярное разложение
        V, D, U_t = np.linalg.svd(X_centered, full_matrices=False)

        # Главные компоненты - собственные вектора X^T @ X
        # декоррелирующее преобразование Карунена-Лоэва.
        self.components_ = U_t

        # Объясненная дисперсия
        self.singular_values_ = D
        self.explained_variance_ = (D ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()
        self._select_n_components()

        return self

    def _select_n_components(self):
        if self.n_components is not None:
            if self.components_ is not None:
                self.components_ = self.components_[:self.n_components]
            if self.explained_variance_ is not None:
                self.explained_variance_ = self.explained_variance_[:self.n_components]
            if self.explained_variance_ratio_ is not None:
                self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components]
            if self.singular_values_ is not None:
                self.singular_values_ = self.singular_values_[:self.n_components]

    def apply_n_components(self, n_components: int):
        if self.components_ is None:
            self.n_components = n_components
            return
        elif self.n_components is None:
            if self.components_.shape[0] <= n_components:
                self.n_components = n_components
            else:
                raise ValueError(f"n_components should be bigger then {self.components_.shape[0]}")
        elif n_components <= self.n_components:
            self.n_components = n_components
        else:
            raise ValueError(f"Can't set n_components bigger then previous {self.n_components}")
        self._select_n_components()

    def transform(self, X):
        """Преобразование данных в пространство главных компонент"""
        return np.dot(X - self.mean_, self.components_.T)

    def fit_transform(self, X):
        """Обучение и преобразование за один шаг"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """Обратное преобразование в исходное пространство"""
        return np.dot(X_transformed, self.components_) + self.mean_

    def get_cumulative_variance(self):
        """Кумулятивная объясненная дисперсия"""
        return np.cumsum(self.explained_variance_ratio_)
