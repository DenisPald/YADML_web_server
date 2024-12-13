import os
import torch

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

