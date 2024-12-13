import os
import asyncio
import asyncssh
import json

from dataset_utils import split_dataset
from ssh_utils import distribute_files_to_nodes, collect_remote_models
from model_utils import aggregate_models, distribute_global_model


async def execute_remote_training(ssh_config: dict, script_path: str, args: dict[str, str]):

    cmd = f"python3 {script_path} " + " ".join([f"--{k} {v}" for k, v in args.items()])
    host = ssh_config['hostname']
    port = ssh_config['port']
    username = ssh_config.get('username', 'root')
    password = ssh_config.get('password', None)

    print(f"Running training on node {host}...")
    try:
        async with asyncssh.connect(
            host=host,
            port=port,
            username=username,
            password=password,
            known_hosts=None
        ) as conn:
            result = await conn.run(cmd, check=False)
            print(f"{host} STDOUT: {result.stdout}")
            print(f"{host} STDERR: {result.stderr}")
            if result.exit_status != 0:
                print(f"Training on {host} failed with exit code {result.exit_status}.")
                return -1
    except Exception as e:
        print(f"Failed to execute remote training on {host}: {e}")
        return -1


def load_config(config_path="conf.json") -> dict:
    """
    Загружает конфигурацию узлов из файла JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    # Валидация структуры данных
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a dict of params.")

    if not isinstance(config['nodes'], list):
        raise ValueError("Configuration file must contain a dict of params.")

    return config


async def main():
    dataset_path = "data/mnist.pt"
    output_dir = "splits/"
    n_splits = 2

    config = load_config()
    nodes = config['nodes']

    remote_dir = "/app/dataset_parts"
    remote_script_path = "/app/train.py"
    remote_model_dir = "/app/models"
    local_model_dir = "local_models/"
    aggregated_model_path = "final_model.pt"

    global_epochs = 4

    # Разделение датасета на части
    split_files = split_dataset(dataset_path, n_splits, output_dir)

    # Распределение частей датасета по узлам (async)
    await distribute_files_to_nodes(split_files, nodes, remote_dir)

    for global_epoch in range(global_epochs):
        print(f"=== Global Epoch {global_epoch}/{global_epochs} ===")

        # Запуск обучения на узлах
        tasks = []
        for i, config in enumerate(nodes):
            args = {
                "data_path": f"{remote_dir}/split_{i}.pt",
                "epochs": "5",  # Количество локальных эпох на узле
                "batch_size": "32",
                "model_save_path": f"{remote_model_dir}/model_{global_epoch}.pt"
            }
            tasks.append(execute_remote_training(config, remote_script_path, args))
        await asyncio.gather(*tasks)

        # Сбор моделей с узлов
        node_models = await collect_remote_models(nodes, remote_model_dir, f"model_{global_epoch}.pt", local_model_dir)

        if not node_models:
            print("No node models were collected. Skipping aggregation for this round.")
            continue

        # Агрегация моделей
        aggregate_models(node_models, aggregated_model_path, input_dim=28 * 28, output_dim=10)

        # Распространение глобальной модели на узлы
        await distribute_global_model(aggregated_model_path, nodes, remote_model_dir)

    print(f"Финальная модель сохранена в {aggregated_model_path}")


if __name__ == "__main__":
    asyncio.run(main())

