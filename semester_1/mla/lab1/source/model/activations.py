from typing import Optional, Union
import numpy as np


class ReverseSigmoid:
    def __init__(self, clip_limit: Optional[Union[int, float]] = 250):
        self.clip_limit = -clip_limit if clip_limit is not None and clip_limit < 0 else clip_limit

    def call(self, x) -> np.ndarray:
        x = np.clip(x, -self.clip_limit, self.clip_limit) if self.clip_limit else x
        return 2 / (1 + np.exp(x))

    def derivative(self, out) -> np.ndarray:
        return out * (out/2 - 1)

    def __call__(self, x):
        return self.call(x)


class LogActivation:
    def __init__(self, clip_limit: Optional[Union[int, float]] = 250):
        self.clip_limit = -clip_limit if clip_limit is not None and clip_limit < 0 else clip_limit

    def call(self, x) -> np.ndarray:
        x = np.clip(x, -self.clip_limit, self.clip_limit) if self.clip_limit else x
        return np.log2(1 + np.exp(-x))

    def derivative(self, x) -> np.ndarray:
        x = np.clip(x, -self.clip_limit, self.clip_limit) if self.clip_limit else x
        return (1/(1 + np.exp(-x)) - 1) / np.log(2)

    def __call__(self, x):
        return self.call(x)