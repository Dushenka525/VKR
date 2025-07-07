"""
Конфигурации и модели данных проекта
"""
from dataclasses import dataclass
from typing import Union


@dataclass
class DataConfig:
    raw_data_path: str
    normalized_data_path: str
    train_data_path: str
    val_data_path: str
    val_size: float
    params_ranges: dict[str, list[float, float]]


@dataclass
class BaselineTrainConfig:
    train_data_path: str
    val_data_path: str
    normalized_data_path: str
    # model: dict[str, Union[int, float, str]]
    experiment_name: str
    model_info_dir: str


@dataclass
class ModelConfig:
    type_model: str
    parameters: dict[str, Union[int, float]]
    model_info_dir: str


@dataclass
class NnTrainConfig:
    artifact_dir: str
    model_dir: str


@dataclass
class ALTrainConfig:
    register_model_name: str
    model_info_dir: str
    query_strategy_type: str
    model_type: str
    model_params: dict[str, Union[int, float]]
    n_start_points: int
    estimation_step: int
    experiment_name: str




@dataclass
class ALTrainConfig2:
    model_info_dir: str
    train_data_path: str
    val_data_path: str
    query_strategy_type: str
    model_type: str
    model_name: str
    n_start_points: int
    n_query: int
    estimation_step: int
    experiment_name: str


@dataclass
class VisualizationConfig:
    artifact_dir: str
    estimation_info_filename: str
    experiment_names: list[str]
    name_encoder: dict[str, str]
    train_data_dim: int


@dataclass
class MLflowConfig:
    uri: str