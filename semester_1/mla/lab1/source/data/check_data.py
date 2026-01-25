from typing import List

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

IMAGES_DIR = "images/"


def _save_and_close(img_name: str):
    img_path = os.path.join(IMAGES_DIR, img_name)
    plt.savefig(img_path)
    plt.close('all')


def plot_target(df: pd.DataFrame, target_col: str):
    plt.figure(figsize=(15, 8))
    df[target_col].value_counts().plot.bar(rot=0)
    plt.title(f"{target_col} values distribution")

    _save_and_close(f"{target_col}_distribution.png")


def show_describe(df: pd.DataFrame, features_type: str) -> pd.DataFrame:
    if features_type not in ["numeric", "categorical"]:
        raise ValueError("Features type should be 'numeric' or 'categorical'")

    df_type = df.select_dtypes(include='number' if features_type == "numeric" else "object")
    df_stats = df_type.describe().T

    cols = df_stats.index
    if 'unique' not in df_stats:
        df_stats['unique'] = df_type[cols].nunique()
    df_stats["nan"] = df_type[cols].isna().sum()
    df_stats["nan %"] = df_stats["nan"] / df_type.shape[0]

    print(f"{features_type.title()} features describe:")
    print(df_stats)

    return df_stats


def plot_mutual_distribution(df: pd.DataFrame, check_mutual_cols: List[str], check_col: str, target_col: str):
    plt.figure(figsize=(20, 8))

    for i, mut_col in enumerate(check_mutual_cols):
        plt.subplot(1, len(check_mutual_cols), i + 1)

        for trg, clr in zip([-1, 1], ['b', 'r']):
            plt.scatter(
                df[df[target_col] == trg][mut_col],
                df[df[target_col] == trg][check_col],
                c=clr,
                label=f"Target: {trg}"
            )
        plt.xlabel(mut_col)
        plt.ylabel(check_col)
        plt.legend()

    _save_and_close(f"Mutual_distribution_with_{check_col}.png")


def plot_density(df: pd.DataFrame, features_type: str):
    if features_type not in ["numeric", "categorical"]:
        raise ValueError("Features type should be 'numeric' or 'categorical'")

    df_type = df.select_dtypes(include='number' if features_type == "numeric" else "object")
    cnt_cols = len(df_type.columns)
    _rows = int(np.ceil(np.sqrt(cnt_cols)))
    _cols = int(np.ceil(cnt_cols / _rows))

    plt.figure(figsize=(20, 15))

    for i, col in enumerate(df_type.columns):
        plt.subplot(_rows, _cols, i + 1)
        sns.histplot(df[col], kde=features_type == "numeric", stat="density")

    plt.tight_layout()
    _save_and_close(f"Density_{features_type}_features.png")

