from typing import Optional
import time

from .load_data import load_data
from .process_data import create_binary_target, encode_features, train_test_split


def run_data_pipeline(
    save_path: Optional[str] = None, with_plotting: bool = True, return_split: bool = True,
    test_size=0.3, random_seed: Optional[int] = None
):
    """
    Run all data preprocessing pipeline:
        1. loading data
        2. plotting info (if with_plotting = True),
        3. Preprocessing and encoding
        4. Splitting to train and test (if return_split = True)

    :param save_path: Optional[str] = None
        Path to save encoded data, data will be saved in csv format
        If None, encoded data not saving
    :param with_plotting: bool = True
        If set to True, will plot data statistics images and save to images/ dir
    :param return_split: bool = True
        Whether return encoded data (False) or splitted to train and test (True)
    :param test_size: float = 0.3
        If return_split set to True, defines test_size proportion
    :param random_seed: Optional[int] = None
        If return_split set to True, given random_seed will be used for splitting

    :return: pd.DataFrame or tuple of 4 np.ndarray depending on `return_split` option
    """
    print("Running data pipeline...")
    start_time = time.time()

    df = load_data()
    df = create_binary_target(df, drop_previous_target=not with_plotting)

    if with_plotting:
        # Import module here because it's not needed when with_plotting option set to False
        # Gives speed up in modules importing
        from .check_data import plot_target, show_describe, plot_mutual_distribution, plot_density

        # Plot previous target and then drop it, new target based on this target
        plot_target(df, 'Obesity')
        df.drop('Obesity', axis=1, inplace=True)

        # Plot new target
        plot_target(df, 'target')
        # Show describe and density by each features group
        for features_type in ["numeric", "categorical"]:
            show_describe(df, features_type=features_type)
            plot_density(df, features_type=features_type)

        # Weight can be most important feature, check it's mutual distributions
        plot_mutual_distribution(
            df, check_mutual_cols=['Age', 'Height'],
            check_col="Weight", target_col="target"
        )

    df = encode_features(df)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    if return_split:
        result = train_test_split(df, test_size=test_size, random_seed=random_seed)
    else:
        result = df

    print(f"Data pipeline finished in {time.time() - start_time:.2f} sec")
    return result
