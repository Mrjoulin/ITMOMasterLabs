from typing import Union
import numpy as np


class Kernel:
    KERNELS = ('linear', 'rbf', 'poly')

    def __init__(self, name: str, gamma: Union[float, str], degree=3):
        self.name = name.lower()
        self.gamma = gamma
        self.degree = degree

        if self.name not in self.KERNELS:
            raise ValueError(f"Неподдерживаемое ядро: {name}. Поддерживаемые ядра: {self.KERNELS}")

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)

    def rbf_kernel(self, x1, x2):
        if isinstance(self.gamma, str):
            raise ValueError("Gamma not set, use .set_gamma(value) to set correct value")

        if x1.ndim == 1 and x2.ndim == 1:
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif x1.ndim == 1:
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2, axis=1) ** 2)
        elif x2.ndim == 1:
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2, axis=0) ** 2)
        else:
            return np.exp(-self.gamma * np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], axis=2) ** 2)

    def polynomial_kernel(self, x1, x2):
        return (np.dot(x1, x2.T) + 1) ** int(self.degree)

    def get_kernel(self, x1, x2):
        if self.name == 'linear':
            return self.linear_kernel(x1, x2)
        elif self.name == 'rbf':
            return self.rbf_kernel(x1, x2)
        elif self.name == 'poly':
            return self.polynomial_kernel(x1, x2)
        else:
            raise ValueError(f"Неподдерживаемое ядро: {self.name}. Поддерживаемые ядра: {self.KERNELS}")

    def __call__(self, x1, x2):
        return self.get_kernel(x1, x2)

    def __str__(self):
        if self.name == "rbf":
            return f"RBF(gamma={self.gamma:.3f})"
        elif self.name == "poly":
            return f"Poly(degree={self.degree})"
        return self.name.title()
