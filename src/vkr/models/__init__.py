"""
Описание классов моделей, стратегий обучения и т.д.
"""
import numpy as np
import torch
import gpytorch
import pandas as pd
import wandb
import os
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, r2_score
from typing import Union, Optional
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel, AdditiveKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from kan import KAN, create_dataset_from_data
from sklearn.base import BaseEstimator, RegressorMixin
import dill  # Альтернатива pickle для сложных объектов
import cloudpickle
from pathlib import Path



class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = (
            ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])) +
            ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class KANWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, width=[7,7,7,1], grid=5, k=3, seed=42, lr=1e-3, epochs=100):
        """
        Обертка для KAN с исправленной сериализацией.
        """
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self.model = None  # Будет инициализирован в fit()
        
    def _init_model(self, input_dim):
        """Отложенная инициализация модели"""
        if self.model is None:
            torch.manual_seed(self.seed)
            self.model = KAN(width=[input_dim] + self.width[1:], 
                            grid=self.grid, 
                            k=self.k)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        # Преобразование данных
        X = torch.tensor(X.values if hasattr(X, 'values') else X, 
                        dtype=torch.float32)
        y = torch.tensor(y.values if hasattr(y, 'values') else y, 
                        dtype=torch.float32)
        
        # Инициализация модели с правильной размерностью входа
        self._init_model(X.shape[1])
        
        # Обучение
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = torch.nn.functional.mse_loss(outputs, y.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            
        return self

    def predict(self, X):
        X = torch.tensor(X.values if hasattr(X, 'values') else X, 
                        dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).numpy().flatten()

    def save(self, path):
        """Сохранение с использованием dill"""
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump({
                'state_dict': self.model.state_dict(),
                'width': self.width,
                'grid': self.grid,
                'k': self.k,
                'seed': self.seed
            }, f)

    @classmethod
    def load(cls, path):
        """Загрузка с использованием dill"""
        with open(path, 'rb') as f:
            data = dill.load(f)
        
        wrapper = cls(width=data['width'], 
                     grid=data['grid'],
                     k=data['k'],
                     seed=data['seed'])
        wrapper._init_model(data['width'][0])
        wrapper.model.load_state_dict(data['state_dict'])
        return wrapper
    

class SerializableKANWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, width=[7,7,7,1], grid=5, k=3, seed=42, lr=1e-3, epochs=100):
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr
        self.epochs = epochs
        self._model = None
        self._optimizer = None
    
    def _convert_to_tensor(self, data):
        """Конвертирует pandas DataFrame/Series в torch.Tensor"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.float()
    
    def _init_model(self, input_dim):
        if self._model is None:
            torch.manual_seed(self.seed)
            self._model = KAN(width=[input_dim] + self.width[1:], grid=self.grid, k=self.k)
            self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
    
    def fit(self, X, y):
        # Конвертация данных
        X_tensor = self._convert_to_tensor(X)
        y_tensor = self._convert_to_tensor(y)
        
        # Инициализация модели
        self._init_model(X_tensor.shape[1])
        
        # Обучение
        self._model.train()
        for epoch in range(self.epochs):
            self._optimizer.zero_grad()
            outputs = self._model(X_tensor)
            loss = torch.nn.functional.mse_loss(outputs, y_tensor.unsqueeze(1))
            loss.backward()
            self._optimizer.step()
        return self
    
    def predict(self, X):
        X_tensor = self._convert_to_tensor(X)
        self._model.eval()
        with torch.no_grad():
            return self._model(X_tensor).numpy().flatten()
    
    def save(self, path):
        """Специальный метод сохранения для KAN"""
        torch.save({
            'width': self.width,
            'grid': self.grid,
            'k': self.k,
            'state_dict': self._model.state_dict()
        }, path)
    
    @classmethod
    def load(cls, path):
        """Специальный метод загрузки для KAN"""
        data = torch.load(path)
        wrapper = cls(width=data['width'], grid=data['grid'], k=data['k'])
        wrapper._init_model(data['width'][0])
        wrapper._model.load_state_dict(data['state_dict'])
        return wrapper

class GPRegressor:
    def __init__(self, nu=1.5, lr=0.1, n_epochs=100):
        self.nu = nu
        self.lr = lr
        self.n_epochs = n_epochs
        self.model = None
        self.likelihood = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _ensure_tensor(self, data):
        """Конвертирует данные в torch.Tensor и переносит на нужное устройство"""
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if isinstance(data, torch.Tensor):
            return data.float().to(self.device)
        raise ValueError(f"Unsupported data type: {type(data)}")

    def fit(self, train_X, train_y):
        
        train_X = self._ensure_tensor(train_X)
        train_y = self._ensure_tensor(train_y)
        
        if train_X.dim() == 1:
            train_X = train_X.unsqueeze(-1)
        if train_y.dim() > 1:
            train_y = train_y.squeeze()
        
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = GPRegressionModel(train_X, train_y, self.likelihood).to(self.device)
        
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            output = self.model(train_X)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
        
        return self

    def predict(self, X, return_std=False):
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
            
        X = self._ensure_tensor(X)
        if X.dim() == 1:
            X = X.unsqueeze(-1)
        
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X))
            mean = observed_pred.mean.cpu().numpy()
            
            if return_std:
                std = observed_pred.stddev.cpu().numpy()
                return mean, std
            return mean


class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = (
            ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=train_x.shape[1])) +
            ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# --- Обёртка для одной модели ---
class GPCommitteeModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    



def qbc(committee, X_sample):

    tx = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True)
    preds = []

    for model, likelihood in committee:
        model.eval()
        likelihood.eval()
        x_query_t = tx.reshape(1, -1)
        mean = model(x_query_t).mean
        preds.append(mean)

    preds_tensor = torch.stack(preds)  
    f_avg = preds_tensor.mean(dim=0)
    loss = torch.var(preds_tensor - f_avg, dim=0).mean()
    tx.grad = None  
    loss.backward()

    gradients = (
        tx.grad.detach().numpy() if tx.grad is not None else np.zeros_like(X_sample)
    )
    return loss.item(), gradients

# Стратегия отбора NA-QBC
def NA_query_strategy(comittee, X_sample, X_pool):
    _, grads = qbc(comittee, X_sample)  # (N, M) -> N
    grads = [torch.tensor(row, dtype=torch.float32).unsqueeze(0) for row in grads]
    grads = torch.tensor(grads, dtype=torch.float32).numpy()
    sign = np.sign(grads)
    for ind in range(len(sign)):
        step = 0.01
        x_gen = X_sample[ind] + step * sign[ind]
        x_gen = np.clip(x_gen, 0.0, 1.0)
        dists = np.linalg.norm(X_pool - x_gen, axis=1)
        idx = np.argmin(dists)
    return idx, grads

#Функция возвращает только значение qbc
def qbc_for_pool(committee, X_sample):

    tx = torch.tensor(X_sample, dtype=torch.float32)

    preds = []
    for model, likelihood in committee:
        model.eval()
        likelihood.eval()
        x_query_t = tx.reshape(1, -1)
        mean = model(x_query_t).mean
        preds.append(mean)


    preds_tensor = torch.stack(preds)  # [n_models, n_samples]
    f_avg = preds_tensor.mean(dim=0)
    loss = torch.var(preds_tensor - f_avg, dim=0).mean()
    return loss


# Стратегия отбора типа pool-based
def pool_based_qbc(committee, X_pool, n_instances=7):

    train_idx = np.random.choice(range(X_pool.shape[0]), size=21, replace=False)
    X_init = X_pool[train_idx]
    
    uncertainties = []
    for x in X_init:
        loss = qbc_for_pool(committee, np.array([x]))
        uncertainties.append(loss.detach().numpy())  # Добавляем .detach() для преобразования в numpy
    
    uncertainties_np = np.array(uncertainties)
    sorted_indices = np.argsort(uncertainties_np)[::-1]
    selected_init_indices = sorted_indices[:n_instances]
    selected_pool_indices = train_idx[selected_init_indices]
    

    
    return selected_pool_indices, X_pool[selected_pool_indices]





class MetricsLogger:
    def __init__(self, save_dir: str = "experiments", name: str='model', wandb_project: Optional[str] = None):
        """
        Инициализация логгера метрик
        
        Args:
            save_dir: Папка для сохранения CSV
            wandb_project: Имя проекта Weights & Biases (если None, логирование в wandb не производится)
        """
        self.save_dir = save_dir
        self.wandb_project = wandb_project
        self.list_r2 = []
        self.list_mae = []
        self.list_iter = []
        
        # Создаем папку для сохранения
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Инициализируем wandb если указан проект
        if self.wandb_project:
            wandb.init(project=self.wandb_project,
                       name=name)

    def log_metrics(self, y_true, y_pred, name_csv, iteration: int):
        """
        Логирует метрики и сохраняет их
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            iteration: Номер итерации
        """
        # Вычисляем метрики
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Сохраняем в списки
        self.list_r2.append(r2)
        self.list_mae.append(mae)
        self.list_iter.append(iteration)
        
        # Сохраняем в CSV
        self._save_to_csv(name_csv)
        
        # Логируем в wandb если инициализирован
        if self.wandb_project and wandb.run is not None:
            wandb.log({
                "iteration": iteration,
                "MAE": mae,
                "R2": r2
            })
        
        return mae, r2

    def _save_to_csv(self, name):
        """Сохраняет метрики в CSV файл"""
        metrics_df = pd.DataFrame({
            'iteration': self.list_iter,
            'MAE': self.list_mae,
            'R2': self.list_r2
        })
        
        csv_path = os.path.join(self.save_dir, name)
        metrics_df.to_csv(csv_path, index=False)

    def finalize(self):
        """Завершает логирование (для wandb)"""
        if self.wandb_project and wandb.run is not None:
            wandb.finish()






def baseline_models(
    gpr_default_params: dict[str, Union[int, float]],
    # rf_default_params: dict[str, Union[int, float]],
    # kan_default_params: dict[str, Union[int, float]],
) -> dict[
    str,
    Union[
        SGDRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
        GPRegressor,
        KAN,
    ],
]:
    base_gpr = GPRegressor(**gpr_default_params)
    # base_kan = KAN(**kan_default_params)
    # base_gbr = GradientBoostingRegressor(**gb_default_params)
    # base_rf = GradientBoostingRegressor(**rf_default_params)
    return {
        "linreg": SGDRegressor(),
        "gboost": GradientBoostingRegressor(),
        "random_forest": RandomForestRegressor(),
        "gpr": base_gpr,
        "kan": SerializableKANWrapper(),
    }