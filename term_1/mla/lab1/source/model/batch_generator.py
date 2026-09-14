from typing import Optional
import numpy as np


class BatchGenerator:
    def __init__(
        self, X: np.ndarray, y: np.ndarray,
        batch_size: int = 32, method: str = 'margin',
        random_seed: Optional[int] = None
    ):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.method = method
        self.dataset_size = X.shape[0]
        self._rng = np.random.default_rng(random_seed)

    def _generator_from_indexes(self, shuffle_indx: np.ndarray):
        batch_cnt = int(np.ceil(self.dataset_size / self.batch_size))
        for batch_indx in range(batch_cnt):
            cur_shuffle_indx = shuffle_indx[batch_indx * self.batch_size:(batch_indx + 1) * self.batch_size]
            yield batch_indx, self.X[cur_shuffle_indx], self.y[cur_shuffle_indx]

    def margin_batches(self, y_pred: np.ndarray):
        # Calculate margins as g(x, w) * y
        margins = y_pred * self.y
        # Ge absolute margins to get probabilities
        margin_abs = np.abs(margins)
        margin_abs[margin_abs - 1e-3 < 0] = 1e-3
        # Get probabilities
        margin_prob = 1 / margin_abs
        margin_prob /= margin_prob.sum()

        shuffle_indx = self._rng.choice(self.dataset_size, size=self.dataset_size, replace=False, p=margin_prob)
        return self._generator_from_indexes(shuffle_indx)

    def random_batches(self):
        shuffle_indx = self._rng.choice(self.dataset_size, size=self.dataset_size, replace=False)
        return self._generator_from_indexes(shuffle_indx)

    def batches(
        self, y_pred: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None, method: Optional[str] = None
    ):
        if batch_size is not None:
            self.batch_size = batch_size
        if method is None:
            method = self.method

        if method == 'margin':
            if y_pred is None:
                raise ValueError("For margin-based batches selection provide y_pred!")
            return self.margin_batches(y_pred)
        else:
            return self.random_batches()
