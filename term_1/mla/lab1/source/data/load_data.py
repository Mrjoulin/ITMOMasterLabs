import kagglehub
import pandas as pd

DATASET_NAME = 'ruchikakumbhar/obesity-prediction'
DATASET_PATH = 'Obesity prediction.csv'


def load_data() -> pd.DataFrame:
    df = kagglehub.dataset_load(
        kagglehub.KaggleDatasetAdapter.PANDAS,
        DATASET_NAME,
        DATASET_PATH
    )
    return df


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df
