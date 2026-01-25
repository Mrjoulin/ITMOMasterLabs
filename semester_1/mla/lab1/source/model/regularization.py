import numpy as np


class Regularization:
    def __init__(self, reg_coef: float):
        self.reg_coef = np.float64(reg_coef)

    def calc(self, weights: np.ndarray):
        raise NotImplementedError("Regularization calculation not implemented")

    def derivative(self, weights: np.ndarray):
        raise NotImplementedError("Regularization derivative not implemented")

    def __call__(self, weights: np.ndarray):
        return self.calc(weights)


class L2Regularization(Regularization):
    def calc(self, weights: np.ndarray):
        return self.reg_coef * ((weights ** 2).sum())

    def derivative(self, weights: np.ndarray):
        return 2 * self.reg_coef * weights
