import paramiko
import numpy as np
import torch
from torch import nn

from model import LeNet


def distribute_global_model(global_model_path: str, nodes: list[dict], remote_model_dir: str):
    """
    Распространяет глобальную модель на все рабочие узлы.
    """
    for config in nodes:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(**config)
        except Exception:
            nodes.remove(config)
            continue;

        # Отправляем глобальную модель через SFTP
        with ssh.open_sftp() as sftp:
            remote_model_path = f"{remote_model_dir}/global_model.pt"
            sftp.put(global_model_path, remote_model_path)



def aggregate_models(model_paths: list, output_path: str, input_dim: int, output_dim: int):
    aggregated_model = LeNet()
    aggregated_state_dict = None
    for path in model_paths:
        model_state_dict = torch.load(path, weights_only=True)
        if aggregated_state_dict is None:
            aggregated_state_dict = model_state_dict
        else:
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += model_state_dict[key]

    for key in aggregated_state_dict:
        aggregated_state_dict[key] /= len(model_paths)

    aggregated_model.load_state_dict(aggregated_state_dict)
    torch.save(aggregated_model.state_dict(), output_path)


"""
Различные методы объединения весов модели в зависимости от класса задач
    
TODO: Интегрировать выбор метода для пользователя
"""

def federated_averaging(weights: list, data_sizes: list) -> list:
    total_data = sum(data_sizes)
    aggregated_weights = [sum(w * n for w, n in zip(weight_layer, data_sizes)) / total_data
                          for weight_layer in zip(*weights)]
    return aggregated_weights


def simple_average(weights: list) -> list:
    aggregated_weights = [sum(weight_layer) / len(weight_layer) for weight_layer in zip(*weights)]
    return aggregated_weights


def median_aggregation(weights: list) -> list:
    aggregated_weights = [np.median(weight_layer, axis=0) for weight_layer in zip(*weights)]
    return aggregated_weights


def regularized_aggregation(initial_weights, new_weights, alpha=0.1):
    simple_avg = simple_average(new_weights)
    aggregated_weights = [(alpha * w0 + (1 - alpha) * w)
                          for w0, w in zip(initial_weights, simple_avg)]
    return aggregated_weights


def trimmed_mean_aggregation(weights: list, trim_ratio: float = 0.1) -> list:
    trimmed_weights = []
    for weight_layer in zip(*weights):
        sorted_weights = sorted(weight_layer)
        trim_count = int(len(sorted_weights) * trim_ratio)
        trimmed_layer = sorted_weights[trim_count:-trim_count]
        trimmed_weights.append(sum(trimmed_layer) / len(trimmed_layer))
    return trimmed_weights

