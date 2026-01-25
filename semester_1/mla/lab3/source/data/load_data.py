import kagglehub
import pandas as pd

DATASET_NAME = 'uciml/red-wine-quality-cortez-et-al-2009'
DATA_FILENAME = "winequality-red.csv"


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_data() -> pd.DataFrame:
    file_path = kagglehub.dataset_download(DATASET_NAME, path=DATA_FILENAME)
    return load_data_from_csv(file_path)



