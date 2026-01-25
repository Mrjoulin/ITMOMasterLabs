from typing import Optional

import numpy as np
import pandas as pd


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df.drop('Id', axis=1, inplace=True)
    df['Bare_Nuclei'] = pd.to_numeric(df['Bare_Nuclei'], errors='coerce')
    df['Bare_Nuclei'] = df['Bare_Nuclei'].fillna(df['Bare_Nuclei'].mode()[0])
    df.rename({"Class": "target"}, axis=1, inplace=True)
    # Encode targets to -1/1
    df['target'] = df['target'].map({2: -1, 4: 1})
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
