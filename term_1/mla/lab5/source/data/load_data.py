import kagglehub
import pandas as pd
import os

DATASET_NAME = 'taweilo/loan-approval-classification-data'
DATA_FILENAME = "loan_data.csv"


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download(DATASET_NAME)
    file_path = os.path.join(path, DATA_FILENAME)
    return load_data_from_csv(file_path)
