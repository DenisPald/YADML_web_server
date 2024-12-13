# model_utils.py
import torch
from torch import nn
from model import LeNet
import asyncssh
import os

def initialize_weights(model: nn.Module):
    """
    Инициализирует веса модели с использованием инициализации Xavier.

    :param model: Модель PyTorch.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def aggregate_models(model_paths: list, output_path: str, input_dim: int, output_dim: int, aggregation_method:str = "simple_average"):
    """
    Аггрегирует модели с учётом выбранного метода и сохраняет объединённую модель.

    :param model_paths: Список путей к моделям на локальной машине.
    :param output_path: Путь для сохранения объединённой модели.
    :param input_dim: Размер входного слоя (не используется в агрегации, может быть полезен для инициализации модели).
    :param output_dim: Размер выходного слоя (не используется в агрегации, может быть полезен для инициализации модели).
    """
    # Загрузка всех state_dict моделей
    models = []
    for path in model_paths:
        try:
            state_dict = torch.load(path, weights_only=True)
            models.append(state_dict)
        except Exception as e:
            print(f"Ошибка при загрузке модели из {path}: {e}")

    if not models:
        print("Нет загруженных моделей для агрегации.")
        return

    # Валидация соответствия ключей в state_dict
    base_keys = models[0].keys()
    for model in models[1:]:
        if model.keys() != base_keys:
            raise ValueError("Все модели должны иметь одинаковые ключи в state_dict.")

    # Выбор метода агрегации
    if aggregation_method == "simple_average":
        aggregated_state_dict = simple_average(models)
    elif aggregation_method == "median":
        aggregated_state_dict = median_aggregation(models)
    elif aggregation_method == "regularized":
        # Предполагаем, что начальные веса - это веса первой модели
        initial_weights = models[0]
        aggregated_state_dict = regularized_aggregation(initial_weights, models, alpha=0.1)
    elif aggregation_method == "trimmed_mean":
        aggregated_state_dict = trimmed_mean_aggregation(models, trim_ratio=0.1)
    else:
        raise ValueError(f"Неизвестный метод агрегации: {aggregation_method}")

    # Создание и инициализация модели
    aggregated_model = LeNet()
    initialize_weights(aggregated_model)
    aggregated_model.load_state_dict(aggregated_state_dict)

    # Сохранение агрегированной модели
    try:
        torch.save(aggregated_model.state_dict(), output_path)
        print(f"Объединённая модель сохранена в {output_path} с использованием метода '{aggregation_method}'.")
    except Exception as e:
        print(f"Ошибка при сохранении агрегированной модели: {e}")


"""
Различные методы объединения весов модели в зависимости от класса задач

TODO: Интегрировать выбор метода для пользователя
"""

def simple_average(models: list) -> dict:
    """
    Простое усреднение весов моделей.

    :param models: Список state_dict моделей.
    :return: Агрегированный state_dict.
    """
    aggregated_state_dict = {}

    for key in models[0].keys():
        aggregated_state_dict[key] = sum(model[key] for model in models) / len(models)

    return aggregated_state_dict


def median_aggregation(models: list) -> dict:
    """
    Агрегация медианой весов моделей.

    :param models: Список state_dict моделей.
    :return: Агрегированный state_dict.
    """
    aggregated_state_dict = {}
    for key in models[0].keys():
        # Сконкатенируем все тензоры для данного ключа
        stacked = torch.stack([model[key] for model in models], dim=0)
        aggregated_state_dict[key] = torch.median(stacked, dim=0)[0]
    return aggregated_state_dict


def regularized_aggregation(initial_weights: dict, new_weights: list, alpha: float = 0.1) -> dict:
    """
    Регуляризованное аггрегирование весов моделей.

    :param initial_weights: Начальные веса модели.
    :param new_weights: Список state_dict новых моделей.
    :param alpha: Коэффициент регуляризации.
    :return: Агрегированный state_dict.
    """
    aggregated_state_dict = {}
    simple_avg = simple_average(new_weights)

    for key in initial_weights.keys():
        aggregated_state_dict[key] = alpha * initial_weights[key] + (1 - alpha) * simple_avg[key]

    return aggregated_state_dict


def trimmed_mean_aggregation(models: list, trim_ratio: float = 0.1) -> dict:
    """
    Агрегация усечённым средним.

    :param models: Список state_dict моделей.
    :param trim_ratio: Доля данных для усечения.
    :return: Агрегированный state_dict.
    """
    aggregated_state_dict = {}
    n_models = len(models)
    trim_count = int(n_models * trim_ratio)

    for key in models[0].keys():
        # Сконкатенируем все тензоры для данного ключа
        stacked = torch.stack([model[key] for model in models], dim=0)
        # Усечём верхние и нижние trim_count значений
        trimmed = torch.sort(stacked, dim=0)[0][trim_count:-trim_count]
        # Усечённое среднее
        aggregated_state_dict[key] = torch.mean(trimmed, dim=0)

    return aggregated_state_dict


async def distribute_global_model(global_model_path: str, nodes: list[dict], remote_model_dir: str):
    """
    Распространяет глобальную модель на все рабочие узлы.

    :param global_model_path: Путь к объединённой модели на главной ноде.
    :param nodes: Список конфигураций узлов, каждый узел представлен как словарь с параметрами подключения.
    :param remote_model_dir: Папка на узле для сохранения модели.
    """
    for config in nodes:
        host = config['hostname']
        port = config['port']
        username = config.get('username', 'root')
        password = config.get('password', None)

        remote_model_path = f"{remote_model_dir}/global_model.pt"
        try:
            async with asyncssh.connect(
                host=host,
                port=port,
                username=username,
                password=password,
                known_hosts=None
            ) as conn:
                sftp = await conn.start_sftp_client()
                await sftp.put(global_model_path, remote_model_path)
                print(f"Uploaded {global_model_path} to {host}:{remote_model_path}")
        except Exception as e:
            print(f"Failed to distribute global model to {host}: {e}")
