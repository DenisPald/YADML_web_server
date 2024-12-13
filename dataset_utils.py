import torch
from torch.utils.data import random_split, Subset
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from torchvision import transforms
import os
import torch

def split_dataset(dataset_path: str, n_splits: int, output_dir: str, split_method: str = 'random') -> list:
    """
    Делит датасет на n частей и сохраняет их в формате .pt.

    :param dataset_path: Путь к исходному датасету (в формате .pt).
    :param n_splits: Количество частей, на которые нужно разделить.
    :param output_dir: Директория для сохранения частей.
    :return: Список путей к разделённым файлам.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем датасет
    try:
        data = torch.load(dataset_path)
        X, y = data["train_X"], data["train_y"]
    except Exception as e:
        print(f"Ошибка при загрузке датасета: {e}")
        raise

    # Нормализация данных
    X = X.float() / 255.0  # Приведение к типу float и нормализация
    y = y.long()

    if split_method == "random":
        splits = random_split_dataset(X, y, n_splits)
    elif split_method == "stratified":
        splits = stratified_split_dataset(X, y, n_splits)
    elif split_method == "balanced":
        splits = balanced_split_dataset(X, y, n_splits)
    elif split_method == "cluster":
        splits = cluster_based_split_dataset(X, n_clusters=n_splits)
    elif split_method == "bootstrap":
        splits = bootstrap_split_dataset(X, y, n_splits)
    elif split_method == "feature":
        # Пример: разделение по первому признаку
        splits = split_by_feature(X, y, feature_indices=[0])
    else:
        raise ValueError(f"Неизвестный метод разделения: {split_method}")

    split_files = []

    for i, split_idx in enumerate(splits):
        if isinstance(split_idx, dict):
            # Для метода 'feature', который возвращает словарь
            for key, indices in split_idx.items():
                split_X, split_y = X[indices], y[indices]
                split_file = os.path.join(output_dir, f"split_feature_{key}.pt")
                try:
                    torch.save({"train_X": split_X, "train_y": split_y}, split_file)
                    split_files.append(split_file)
                    print(f"Часть {key} сохранена в {split_file}")
                except Exception as e:
                    print(f"Ошибка при сохранении части {key}: {e}")
                    raise
        else:
            # Для остальных методов, которые возвращают список индексов
            split_X, split_y = X[split_idx], y[split_idx]
            split_file = os.path.join(output_dir, f"split_{i}.pt")
            try:
                torch.save({"train_X": split_X, "train_y": split_y}, split_file)
                split_files.append(split_file)
                print(f"Часть {i} сохранена в {split_file}")
            except Exception as e:
                print(f"Ошибка при сохранении части {i}: {e}")
                raise

    return split_files


def random_split_dataset(X: torch.Tensor, y: torch.Tensor, n_splits: int) -> list:
    """
    Случайное разделение датасета на n частей.

    :param X: Тензор признаков.
    :param y: Тензор меток.
    :param n_splits: Количество частей.
    :return: Список индексов для каждой части.
    """
    dataset_size = len(X)
    indices = torch.randperm(dataset_size)
    split_size = dataset_size // n_splits
    splits = [indices[i * split_size:(i + 1) * split_size] for i in range(n_splits)]
    if dataset_size % n_splits != 0:
        splits[-1] = torch.cat((splits[-1], indices[n_splits * split_size:]))
    return splits


def stratified_split_dataset(X: torch.Tensor, y: torch.Tensor, n_splits: int) -> list:
    """
    Стратифицированное разделение датасета на n частей.

    :param X: Тензор признаков.
    :param y: Тензор меток.
    :param n_splits: Количество частей.
    :return: Список индексов для каждой части.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    y_np = y.numpy()

    for _, test_idx in skf.split(X.numpy(), y_np):
        splits.append(torch.tensor(test_idx))

    return splits


def balanced_split_dataset(X: torch.Tensor, y: torch.Tensor, n_splits: int) -> list:
    """
    Балансированное разделение датасета на n частей с использованием RandomUnderSampler.

    :param X: Тензор признаков.
    :param y: Тензор меток.
    :param n_splits: Количество частей.
    :return: Список индексов для каждой части.
    """
    rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    X_res, y_res = rus.fit_resample(X.numpy(), y.numpy())
    X_res = torch.tensor(X_res)
    y_res = torch.tensor(y_res)
    return random_split_dataset(X_res, y_res, n_splits)


def cluster_based_split_dataset(X: torch.Tensor, n_clusters: int) -> list:
    """
    Кластеризованное разделение датасета на n кластеров.

    :param X: Тензор признаков.
    :param n_clusters: Количество кластеров.
    :return: Список индексов для каждой кластера.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X.numpy())
    splits = []
    for i in range(n_clusters):
        split_idx = np.where(clusters == i)[0]
        splits.append(torch.tensor(split_idx))
    return splits


def split_by_feature(X: torch.Tensor, y: torch.Tensor, feature_indices: list) -> dict:
    """
    Разделение датасета по определённым признакам.

    :param X: Тензор признаков.
    :param y: Тензор меток.
    :param feature_indices: Список индексов признаков для разделения.
    :return: Словарь, где ключи — значения признаков, а значения — индексы соответствующих образцов.
    """
    splits = {}
    for idx in feature_indices:
        feature_values = torch.unique(X[:, idx])
        for value in feature_values:
            split_idx = torch.where(X[:, idx] == value)[0]
            splits[f"feature_{idx}_value_{value.item()}"] = split_idx
    return splits


def bootstrap_split_dataset(X: torch.Tensor, y: torch.Tensor, n_splits: int) -> list:
    """
    Бутстрэпное разделение датасета на n частей.

    :param X: Тензор признаков.
    :param y: Тензор меток.
    :param n_splits: Количество частей.
    :return: Список индексов для каждой части.
    """
    splits = []
    for i in range(n_splits):
        split_size = len(X)
        split_idx = torch.randint(0, len(X), (split_size,))
        splits.append(split_idx)
    return splits
