import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
import numpy as np
import torch
import os


def split_dataset(dataset_path: str, n_splits: int, output_dir: str):
    """
    Делит датасет на n частей и сохраняет их в формате .pt.

    :param dataset_path: Путь к исходному датасету (в формате .pt).
    :param n_splits: Количество частей, на которые нужно разделить.
    :param output_dir: Директория для сохранения частей.
    :return: Список путей к разделенным файлам.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем датасет
    data = torch.load(dataset_path, weights_only=True)
    X, y = data["train_X"], data["train_y"]

    # Рассчитываем размер каждого раздела
    split_size = len(X) // n_splits
    split_files = []

    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != n_splits - 1 else len(X)

        split_X, split_y = X[start_idx:end_idx], y[start_idx:end_idx]
        split_file = os.path.join(output_dir, f"split_{i}.pt")

        torch.save({"train_X": split_X, "train_y": split_y}, split_file)
        split_files.append(split_file)

    return split_files


"""
Различные методы деления датасета в зависимости от класса задач
    
TODO: Интегрировать выбор метода для пользователя
"""
def random_split(dataset: pd.DataFrame, n_splits: int) -> list:
    return np.array_split(dataset.sample(frac=1, random_state=42), n_splits)


def stratified_split(dataset: pd.DataFrame, n_splits: int, target_column: str) -> list:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = []
    for _, test_idx in skf.split(dataset, dataset[target_column]):
        splits.append(dataset.iloc[test_idx])
    return splits


def split_by_feature(dataset: pd.DataFrame, feature: str) -> dict:
    return {group: subset for group, subset in dataset.groupby(feature)}


def bootstrap_split(dataset: pd.DataFrame, n_splits: int) -> list:
    return [dataset.sample(frac=1, replace=True, random_state=i) for i in range(n_splits)]


def balanced_split(dataset: pd.DataFrame, target_column: str, n_splits: int) -> list:
    rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)
    dataset_balanced, _ = rus.fit_resample(dataset, dataset[target_column])
    return random_split(dataset_balanced, n_splits)


def time_based_split(dataset: pd.DataFrame, time_column: str, n_splits: int) -> list:
    dataset = dataset.sort_values(by=time_column)
    return np.array_split(dataset, n_splits)


def cluster_based_split(dataset: pd.DataFrame, n_clusters: int) -> list:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dataset['cluster'] = kmeans.fit_predict(dataset)
    return [dataset[dataset['cluster'] == i].drop(columns='cluster') for i in range(n_clusters)]
