"""
Скрипты для загрузки и обработки датасетов
"""
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
from src.vkr.conf import DataConfig

@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def make_dataset(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    data_config = DataConfig(**cfg.get("data", {}))
    normalize_features(
        data_config.raw_data_path,
        data_config.params_ranges,
        data_config.normalized_data_path,
    )
    split_data(
        data_config.normalized_data_path,
        data_config.train_data_path,
        data_config.val_data_path,
        data_config.test_data_path,
        data_config.val_size,
        data_config.test_size,
    )
    return None

def normalize_features(
    raw_data_path: str,
    params_ranges: dict[str, list[float]],
    normalized_data_path: str,
) -> None:
    df = pd.read_csv(raw_data_path)
    def processing(df):
        for i in df.columns:
            if df[i].dtype is not np.float64:
                df[i] = df[i].astype(np.float64)
        scaler = MinMaxScaler()
        # Fit and transform the data
        d_f = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return d_f
    df = processing(df)
    df.to_csv(normalized_data_path, index=False)
    return None


def split_data(
    raw_data_path: str,
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    val_size: float,
    test_size: float,
    random_state: int = 1234,
) -> None:
    df = pd.read_csv(raw_data_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
    )
    train_df.to_csv(train_data_path, index=False)
    val_df.to_csv(val_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)
    return None


if __name__ == "__main__":
    make_dataset()

# with open('configs/data/merged_gen_v4.yaml', 'r') as file:
#     templates = yaml.safe_load(file)

# print(templates)
# print(templates['raw_data_path'])

# df = pd.read_csv(templates['raw_data_path'])

# print(df.columns)
# for i in df.columns:
#     if df[i].dtype is not np.float64:
#         df[i] = df[i].astype(np.float64)

# print(df)
# scaler = MinMaxScaler()
# # Fit and transform the data
# df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# print(df)