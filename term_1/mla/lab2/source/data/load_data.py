import os
import kagglehub
import pandas as pd

DATASET_NAME = 'zzero0/uci-breast-cancer-wisconsin-original'
DATASET_PATH = './dataset'
DATA_FILENAME = "breast-cancer-wisconsin.data.txt"


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def load_data_from_txt(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None)
    col_names = [
        'Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion',
        'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class'
    ]
    df.columns = col_names
    return df


def load_data() -> pd.DataFrame:
    file_path = kagglehub.dataset_download(DATASET_NAME, path=DATA_FILENAME)
    return load_data_from_txt(file_path)



