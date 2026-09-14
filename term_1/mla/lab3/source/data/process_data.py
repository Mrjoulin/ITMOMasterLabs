from typing import Optional

import numpy as np
import pandas as pd


class StandardScaler:
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray):
        self._mean = X.mean(axis=0, keepdims=True)
        self._std = X.std(axis=0, keepdims=True)

    def transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return (X - self._mean) / self._std

    def fit_transform(self, X: np.ndarray):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScaler wasn't fitted")
        return X * self._std + self._mean


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encode targets to -1/1
    df['target'] = -1
    df.loc[df['quality'] >= 6, 'target'] = 1
    df = df.drop('quality', axis=1)

    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.3, random_seed: Optional[int] = None):
    rng = np.random.default_rng(random_seed)

    # Stratify by target
    targets_probs = df['target'].value_counts(normalize=True)
    probs = df['target'].map(targets_probs)
    probs /= probs.sum()

    rnd_indexes = rng.choice(df.shape[0], df.shape[0], replace=False, p=probs.to_numpy())

    # Split
    split_lim = round(df.shape[0] * (1 - test_size))

    features_arr = df.drop('target', axis=1).to_numpy()
    target_arr = df['target'].to_numpy()

    X_train, X_test = features_arr[rnd_indexes[:split_lim]], features_arr[rnd_indexes[split_lim:]]
    y_train, y_test = target_arr[rnd_indexes[:split_lim]], target_arr[rnd_indexes[split_lim:]]

    return X_train, X_test, y_train, y_test
