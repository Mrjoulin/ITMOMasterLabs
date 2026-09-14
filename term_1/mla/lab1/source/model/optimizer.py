import numpy as np


class MomentumOptimizer:
    def __init__(self, learning_rate: float = 1e-4, momentum_betta: float = 0.5):
        self.lr = learning_rate
        self.betta = momentum_betta
        self.prev_grads = {}

    def apply_gradients(self, weights_name: str, weights: np.ndarray, grads: np.ndarray):
        if weights_name in self.prev_grads:
            self.prev_grads[weights_name] = self.prev_grads[weights_name] * self.betta + self.lr * grads
        else:
            self.prev_grads[weights_name] = self.lr * grads
        return weights - self.prev_grads[weights_name]
