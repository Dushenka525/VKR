"""
Скрипты для обучения моделей
"""
import hydra
from pathlib import Path
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import torch
import sys
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
from src.vkr.conf import ModelConfig,BaselineTrainConfig
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from kan import *
from kan import KAN
@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def study_model(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    train_X, train_y = split_x_y(train_config.train_data_path)
    val_X, val_y = split_x_y(train_config.test_data_path)

    if train_config.model['type_model'] == 'gb':
        model = GradientBoostingRegressor(loss = 'absolute_error', learning_rate = train_config.model['learning_rate'],
                                            n_estimators = train_config.model['n_estimators'],max_depth= train_config.model['max_depth'])
        
    elif train_config.model['type_model'] == 'sgr':
        model = SGDRegressor(loss = train_config.model['squared_error'], tol = train_config.model['tol'],
                                            max_iter = train_config.model['max_iter'], learning_rate = train_config.model['learning_rate'])
    
    elif train_config.model['type_model'] == 'rf':
         model = RandomForestRegressor(criterion = 'absolute_error',
                                            n_estimators = train_config.model['n_estimators'],max_depth= train_config.model['max_depth'])
    
    elif train_config.model['type_model'] == 'kan':
        model = KAN(loss = 'absolute_error', learning_rate = train_config.model['learning_rate'],
                                            n_estimators = train_config.model['n_estimators'],max_depth= train_config.model['max_depth'])
        # model.fit(X_train,y_train)
        # y_pred = model.predict(X_val)
        # print(f'Mean Absolute Error: {mean_absolute_error(y_val, y_pred)}')



def split_x_y(path:str) -> np.ndarray: 
    data = pd.read_csv(path)
    train_X = data.iloc[:,:-1]
    train_y = data.iloc[:,-1]
    return train_X, train_y



def create_model_study(train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    model: 
        Union[
            SGDRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            KAN,
        ],
    # model_info_dir: str,
) -> None:

    # for i, (model_name, model) in enumerate(models.items()):
    model = model.fit(train_X, train_y)

    # with mlflow.start_run(run_name=model_name):
    train_preds = model.predict(train_X)
    val_preds = model.predict(val_X)

    train_r2 = r2_score(train_y, train_preds)
    train_mse = mean_squared_error(train_y, train_preds)
    train_mape = mean_absolute_percentage_error(train_y, train_preds)
    train_mae = mean_absolute_error(train_y, train_preds)

    val_r2 = r2_score(val_y, val_preds)
    val_mse = mean_squared_error(val_y, val_preds)
    val_mape = mean_absolute_percentage_error(val_y, val_preds)
    val_mae = mean_absolute_error(val_y, val_preds)

            #     mlflow.log_metrics(
            #         {
            #             "tran_mae": train_mae,
            #             "train_mape": train_mape,
            #             "train_mse": train_mse,
            #             "train_R2": train_r2,
            #             "val_mape": val_mape,
            #             "val_mae": val_mae,
            #             "val_mse": val_mse,
            #             "val_R2": val_r2,
            #         }
            #     )

            #     signature = infer_signature(train_X, train_preds)
            #     model_info = mlflow.sklearn.log_model(
            #         sk_model=model,
            #         artifact_path=f"Baseline {model_name}",
            #         signature=signature,
            #         input_example=train_X,
            #         registered_model_name=model_name,
            #     )

            # model_info_path = str(Path(model_info_dir, f"{model_name}_info.joblib"))
            # dump(model_info, model_info_path)

    return None





if __name__ == "__main__":
    study_model()


