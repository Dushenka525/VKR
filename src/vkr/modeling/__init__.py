"""
Скрипты для обучения моделей
"""
import hydra
from pathlib import Path
import numpy as np
import time
import pandas as pd
import os
from typing import List
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Union
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from joblib import dump
import mlflow
from mlflow.models import infer_signature
import torch
import sys
project_root = Path(__file__).absolute().parents[3]  # До My_VKR
sys.path.insert(0, str(project_root))
from src.vkr.conf import ModelConfig, BaselineTrainConfig,ALTrainConfig
from src.vkr.models import *
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error, r2_score,mean_absolute_percentage_error
import wandb
import joblib


wandb.login()

@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def study_model(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    train_X, train_y = split_x_y(train_config.train_data_path)
    val_X, val_y = split_x_y(train_config.val_data_path)
    test_X, test_y = split_x_y(train_config.test_data_path)
    model_config = ModelConfig(**cfg.get("model", {}))
    models = baseline_models(model_config.parameters)
    print(models,'models')
    create_model_study(train_X, train_y, val_X, val_y, models, model_config.model_info_dir)


def split_x_y(path:str) -> np.ndarray: 
    data = pd.read_csv(path)
    train_X = data.iloc[:,:-1]
    train_y = data.iloc[:,-1]
    return train_X, train_y




def create_model_study(train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    # test_X: np.ndarray,
    # test_y: np.ndarray,
    models: 
        Union[
            SGDRegressor,
            GradientBoostingRegressor,
            RandomForestRegressor,
            # GPRegressionModel,
            GPRegressor,
            SerializableKANWrapper,
        ],
    model_info_dir: str,
    project_name: str = "my_model_comparison",
    ) -> None:
    for model_name, model in models.items():
        timestamp = int(time.time())
        run_name = f"{model_name}_{timestamp}"
        
        with wandb.init(project=project_name, 
                       name=run_name,
                       reinit=True,
                       settings=wandb.Settings(symlink=False),
                       tags=["experiment", model_name, f"version_{timestamp}"]) as run:
            
            print(f"Training {model_name}...")
            
            try:
                # Обучение модели
                model.fit(train_X, train_y)
                
                # Предсказания
                train_preds = model.predict(train_X)
                val_preds = model.predict(val_X)

                # Метрики
                metrics = {
                    "train": {
                        "r2": r2_score(train_y, train_preds),
                        "mse": mean_squared_error(train_y, train_preds),
                        "mape": mean_absolute_percentage_error(train_y, train_preds),
                        "mae": mean_absolute_error(train_y, train_preds)
                    },
                    "val": {
                        "r2": r2_score(val_y, val_preds),
                        "mse": mean_squared_error(val_y, val_preds),
                        "mape": mean_absolute_percentage_error(val_y, val_preds),
                        "mae": mean_absolute_error(val_y, val_preds)
                    }
                }
                
                # Логирование метрик
                wandb.log({
                    "train/mae": metrics["train"]["mae"],
                    "train/mape": metrics["train"]["mape"],
                    "train/mse": metrics["train"]["mse"],
                    "train/r2": metrics["train"]["r2"],
                    "val/mape": metrics["val"]["mape"],
                    "val/mae": metrics["val"]["mae"],
                    "val/mse": metrics["val"]["mse"],
                    "val/r2": metrics["val"]["r2"],
                })
                
                # Сохранение модели (особый случай для KAN)
                model_dir = Path(model_info_dir) / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                if isinstance(model, SerializableKANWrapper):
                    # Для KAN используем специальный метод сохранения
                    model_path = model_dir / f"model_{timestamp}.pt"
                    model.save(model_path)
                    
                    # Создаем артефакт
                    artifact = wandb.Artifact(
                        name=f"{model_name}_model",
                        type="model",
                        description=f"KAN model ({model_name})"
                    )
                    artifact.add_file(str(model_path))
                    run.log_artifact(artifact)
                else:
                    # Для других моделей используем стандартный подход
                    model_path = model_dir / f"model_{timestamp}.joblib"
                    dump(model, model_path)
                    
                    artifact = wandb.Artifact(
                        name=f"{model_name}_model",
                        type="model",
                        description=f"{type(model).__name__} model"
                    )
                    artifact.add_file(str(model_path))
                    run.log_artifact(artifact)
                
                # Сохранение информации о модели
                info_path = model_dir / f"model_info_{timestamp}.json"
                model_info = {
                    "model_name": model_name,
                    "version": timestamp,
                    "metrics": metrics,
                    "model_path": str(model_path)
                }
                dump(model_info, info_path)
                
                info_artifact = wandb.Artifact(
                    name=f"{model_name}_info",
                    type="metadata",
                    description="Model metadata"
                )
                info_artifact.add_file(str(info_path))
                run.log_artifact(info_artifact)
                
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                wandb.log({"error": str(e)})
                continue

    return None




@hydra.main(
    version_base=None, config_path="../../../configs", config_name="config"
)
def active_lear(cfg : DictConfig) -> None:
    OmegaConf.to_yaml(cfg)
    train_config = BaselineTrainConfig(**cfg.get("train", {}))
    models_parametrs = ALTrainConfig(**cfg.get("model", {}))
    X_pool, y_pool = split_x_y(train_config.normalized_data_path)
    np.random.seed(42)
    INIT_SIZE = models_parametrs.n_start_points
    initial_idx = np.random.choice(len(y_pool), size=INIT_SIZE, replace=False)
    X_train, y_train = X_pool.iloc[initial_idx], y_pool.iloc[initial_idx]
    X_pool, y_pool = (
        np.delete(X_pool, initial_idx, axis=0),
        np.delete(y_pool, initial_idx, axis=0),
    )
    if models_parametrs.query_strategy_type == 'qbc':
        process_qbc(X_train, y_train, models_parametrs, X_pool, y_pool)
    elif models_parametrs.query_strategy_type == 'naqbc':
        process_naqbc(X_train, y_train, models_parametrs, X_pool, y_pool)
    elif models_parametrs.query_strategy_type == 'random':
        process_random(X_train, y_train, models_parametrs, X_pool, y_pool)
    

def train_committee(X_train, y_train, config, num_iter=60):
    committee = []
    for params in config.model_params.regressors_params:
        X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
        y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train
        train_x = torch.tensor(X_train, dtype=torch.float32)
        train_y = torch.tensor(y_train, dtype=torch.float32)
        likelihood = GaussianLikelihood()
        kernel = make_kernel(params)
        model = GPCommitteeModel(train_x, train_y, likelihood, kernel)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        mll = ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        for it in range(num_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            if torch.isnan(loss):
                print(f"NaN in model {i}, iter {it}, breaking")
                break
            loss.backward()
            optimizer.step()
        committee.append((model, likelihood))
    return committee


def process_naqbc(X_train, y_train, models_parametrs, X_pool, y_pool):
    # run_name = f"{model_name}_{timestamp}"
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"),  # Папка для сохранения CSV
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  # Имя проекта в wandb (None чтобы отключить)
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step + models_parametrs.n_start_points):
        committee = train_committee(X_train, y_train, models_parametrs)
        best_grad, best_ind = None, None
        for rnd_indx in range(X_train.shape[0]):
            idx, grad = NA_query_strategy(committee, X_train[rnd_indx], X_pool)
            if best_grad is None:
                best_grad = grad
                best_ind = idx
            how_1 = comparison = [abs(g1) > abs(g2) for g1, g2 in zip(grad, best_grad)]
            if len(how_1) > len(best_grad):
                best_grad = grad
                best_ind = idx
        idx = best_ind
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
        )
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")
    metrics_logger.finalize()


def process_qbc(X_train, y_train, models_parametrs, X_pool, y_pool):
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"),  # Папка для сохранения CSV
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  # Имя проекта в wandb (None чтобы отключить)
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step + models_parametrs.n_start_points):
        committee = train_committee(X_train, y_train, models_parametrs)
    
    
        idx, x_dop = pool_based_qbc(committee, X_pool)
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
            
        )
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics_logger.finalize()

def process_random(X_train, y_train, models_parametrs, X_pool, y_pool):
    metrics_logger = MetricsLogger(
        save_dir=Path("C:/Users/ivan/VKR/experiments"),  # Папка для сохранения CSV
        name=models_parametrs.experiment_name,
        wandb_project="Active_learning"  # Имя проекта в wandb (None чтобы отключить)
    )

    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    X_pool = X_pool.values if isinstance(X_pool, pd.DataFrame) else X_pool
    for it in range(models_parametrs.n_start_points, models_parametrs.estimation_step + models_parametrs.n_start_points):
        committee = train_committee(X_train, y_train, models_parametrs)
    
    
        idx = np.random.choice(range(X_pool.shape[0]), size=1, replace=False)
        X_train = np.vstack([X_train, X_pool[idx]])
        y_train = np.append(y_train, y_pool[idx])
        X_pool = np.delete(X_pool, idx, axis=0)
        y_pool = np.delete(y_pool, idx, axis=0)
        print(f"Step {it + 1}: added point {idx}, pool left: {len(X_pool)}")
        preds_list = []

        for model, likelihood in committee:
            model.eval()
            likelihood.eval()
            X_hold = torch.tensor(X_train, dtype=torch.float32)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = model(X_hold)
                y_pred = preds.mean.cpu().numpy().flatten()
                y_true = y_train
            preds_list.append(y_pred)
        y_pred = np.mean(preds_list, axis=0)
        mae, r2 = metrics_logger.log_metrics(
            y_true=y_true,
            y_pred=y_pred,
            name_csv=models_parametrs.experiment_name,
            iteration=it,
            
        )
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"Final MAE: {mae:.4f}, R2: {r2:.4f}")

    metrics_logger.finalize()

def make_kernel(params) -> ScaleKernel:
    # Создаем ядро в зависимости от типа
    if params.kernel_type == 'RBF':
        kernel = ScaleKernel(RBFKernel(ard_num_dims=params.input_dim))
    elif params.kernel_type == 'Matern':
        kernel = ScaleKernel(MaternKernel(nu=params.nu, ard_num_dims=params.input_dim))
    return kernel



if __name__ == "__main__":
    # study_model()
    active_lear()

