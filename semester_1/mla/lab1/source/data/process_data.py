from typing import Optional

import numpy as np
import pandas as pd


def create_binary_target(df: pd.DataFrame, drop_previous_target: bool = False) -> pd.DataFrame:
    # Encode targets:
    # Obesity of any type -> 1
    # Everything else -> 0
    # Target logic: does person have obesity

    df['target'] = -1
    df.loc[df['Obesity'].str.startswith('Obesity'), 'target'] = 1

    if drop_previous_target:
        df.drop('Obesity', axis=1, inplace=True)

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    # Encode binary categorical features, no - 0, yes - 1/female - 0, male - 1
    df['Gender'] = df['Gender'].map({"Female": 0, "Male": 1})

    yes_no_cols = ["family_history", "FAVC", "SMOKE", "SCC"]
    for col in yes_no_cols:
        df[col] = df[col].map({"no": 0, "yes": 1})

    # Encode staged features
    staged_cols = ['CAEC', 'CALC']
    stages_encode = {
        'no': 0.0,
        'Sometimes': 0.5,
        'Frequently': 0.8,
        'Always': 1.0
    }
    for col in staged_cols:
        df[col] = df[col].map(stages_encode)

    # One-hot encode for MTRANS feature
    # Also drop first value (most frequent)
    transport_values = df['MTRANS'].unique()
    top_value = df['MTRANS'].mode().values[0]

    for transport_value in transport_values:
        if transport_value != top_value:
            df[f'MTRANS_{transport_value}'] = (df['MTRANS'] == transport_value).astype(int)

    df.drop('MTRANS', axis=1, inplace=True)

    # MinMax Scale for numeric features
    df_num = df.select_dtypes(include='number')
    df_num_scaled = (df_num - df_num.min()) / (df_num.max() - df_num.min())
    df_num_scaled.drop('target', axis=1, inplace=True)

    df[df_num_scaled.columns] = df_num_scaled
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
