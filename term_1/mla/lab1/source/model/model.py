from typing import Optional, Union, List
import numpy as np

from .batch_generator import BatchGenerator
from .loss import LOSSES, BaseLoss
from .regularization import L2Regularization
from .optimizer import MomentumOptimizer


class LinearClassifier:
    WHEIGHT_INIT_METHODS = ["random", "correlation"]
    BATCH_METHODS = ["random", "margin"]

    def __init__(
            self, weights_init_method: str = "random", batch_method: str = 'margin',
            learning_rate: float = 1e-4, momentum_betta: float = 0.5,
            l2_coef: Optional[float] = 1e-3, loss: Union[str, BaseLoss] = "log_loss", loss_lambda: float = 1e-3
    ):
        self.weights_init_method = LinearClassifier._validate_field(
            weights_init_method, self.WHEIGHT_INIT_METHODS, deafult="random"
        )
        self.batch_method = LinearClassifier._validate_field(
            batch_method, self.BATCH_METHODS, deafult="margin"
        )

        self.optimizer = MomentumOptimizer(learning_rate, momentum_betta)
        self.l2_reg = L2Regularization(l2_coef) if l2_coef is not None else None

        if isinstance(loss, str) and LOSSES.get(loss) is not None:
            self.loss = LOSSES[loss](loss_lambda, self.l2_reg)
        elif isinstance(loss, BaseLoss):
            self.loss = loss
        else:
            raise ValueError("Loss should be string with loss name or loss object")

        self.weights = None
        self.bias = None

    @staticmethod
    def _validate_field(field_value: Optional[str], correct_values: List[str], deafult: Optional[str] = None) -> str:
        if field_value is None:
            if deafult is not None and deafult in correct_values:
                return deafult
            else:
                raise ValueError(f"Field value not given, should be one of: {', '.join(correct_values)}")

        if not isinstance(field_value, str):
            raise ValueError(f"Field value should be string, while {type(field_value)} is given")

        if field_value in correct_values:
            return field_value
        else:
            raise ValueError(f"Incorrect field value, should be one of: {', '.join(correct_values)}")

    def init_weights(self, X: np.ndarray, y: np.ndarray):
        n_features = X.shape[1]

        if self.weights_init_method == 'correlation':
            self.weights = np.zeros((n_features, 1))
            self.bias = np.zeros(1)
            for i in range(n_features):
                feat = X[:, i]
                self.weights[i, 0] = np.dot(y, feat) / np.dot(feat, feat)
        elif self.weights_init_method == 'random':
            norm_scale = 2. / n_features
            self.weights = np.random.normal(scale=norm_scale, size=(n_features, 1))
            self.bias = np.random.normal(scale=norm_scale, size=1)

    def forward(self, X: np.ndarray):
        out = np.dot(X, self.weights) - self.bias
        return out[:, 0]

    def predict(self, X: np.ndarray):
        return np.sign(self.forward(X))

    def train_step(self, X_batch: np.ndarray, y_batch: np.ndarray):
        y_pred = self.forward(X_batch)
        loss = self.loss.get_loss(y_batch, y_pred, self.weights)

        # Calculate gradients
        doutput = self.loss.derivative(y_batch, y_pred)
        if self.l2_reg is not None:
            dweights_reg = self.l2_reg.derivative(self.weights)
        else:
            dweights_reg = 0.0

        dweights = np.dot(X_batch.T, doutput) + dweights_reg
        dbias = np.sum(doutput, axis=0)

        # Apply gradients
        self.weights = self.optimizer.apply_gradients("weights", self.weights, dweights)
        self.bias = self.optimizer.apply_gradients("bias", self.bias, dbias)

        return loss

    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
        epochs: int = 100, batch_size: int = 32, verbose: int = 1, random_seed: Optional[int] = None
    ):
        if self.weights is None:
            self.init_weights(X_train, y_train)

        # Init batch generator
        batch_gen = BatchGenerator(
            X_train, y_train, batch_size=batch_size,
            method=self.batch_method, random_seed=random_seed
        )

        # обучение
        for epoch in range(epochs):
            y_pred = self.forward(X_train) if self.batch_method == "margin" else None

            for batch_indx, X_batch, y_batch in batch_gen.batches(y_pred=y_pred):
                loss = self.train_step(X_batch, y_batch)

                log_message = f"[Epoch {epoch}/{epochs}, Batch {batch_indx}] Loss: {loss:.4f}"
                if verbose > 1:
                    print(f"{log_message}")
                elif verbose > 0:
                    print(f"\r{log_message}", end="")

            if X_test is not None and y_test is not None:
                y_test_pred = self.forward(X_test)
                val_loss = self.loss.val_loss(y_test, y_test_pred)
                val_loss_message = f", Val Loss: {val_loss:.4f}"
            else:
                val_loss_message = ""

            if verbose > 0:
                print(f"\r[Epoch {epoch}/{epochs}] Loss: {self.loss.history[-1]:.4f}{val_loss_message}")

        if X_test is not None and y_test is not None:
            return self.loss.history, self.loss.val_history
        else:
            return self.loss.history
