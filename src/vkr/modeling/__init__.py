"""
Скрипты для обучения моделей
"""
import hydra
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
import sys
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
from src.vkr.conf import ModelConfig,BaselineTrainConfig
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score

@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def study_model(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    df1 = pd.read_csv(train_config.train_data_path)
    X_train = df1.iloc[:,:-1]
    y_train = df1.iloc[:,-1]
    df2 = pd.read_csv(train_config.val_data_path)
    X_val = df1.iloc[:,:-1]
    y_val = df1.iloc[:,-1]
    if train_config.model['type_model'] == 'GradientBoost':
        model = GradientBoostingRegressor(loss = 'absolute_error', learning_rate = train_config.model['learning_rate'],
                                            n_estimators = train_config.model['n_estimators'],max_depth= train_config.model['max_depth'])
        
        model.fit(X_train,y_train)
        y_pred = model.predict(X_val)
        print(f'Mean Absolute Error: {mean_absolute_error(y_val, y_pred)}')
            
if __name__ == "__main__":
    study_model()