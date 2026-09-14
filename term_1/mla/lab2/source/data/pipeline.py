from typing import Optional
import time

from .load_data import load_data
from .process_data import prepare_features, train_test_split


def run_data_pipeline(
    save_path: Optional[str] = None, return_split: bool = True, test_size=0.3, random_seed: Optional[int] = None
):
    """
    Run all data preprocessing pipeline:
        1. loading data
        3. Preprocessing
        4. Splitting to train and test (if return_split = True)

    :param save_path: Optional[str] = None
        Path to save encoded data, data will be saved in csv format
        If None, encoded data not saving
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
    df = prepare_features(df)

    if save_path is not None:
        df.to_csv(save_path, index=False)

    if return_split:
        result = train_test_split(df, test_size=test_size, random_seed=random_seed)
    else:
        result = df

    print(f"Data pipeline finished in {time.time() - start_time:.2f} sec")
    return result
