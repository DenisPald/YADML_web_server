import os
import paramiko
import asyncio
import asyncssh
import json

from dataset_utils import split_dataset
from ssh_utils import distribute_files_to_nodes
from model_utils import aggregate_models, distribute_global_model


async def execute_remote_training(ssh_config: dict, script_path: str, args: dict[str, str]):
    cmd = f"python3 {script_path} " + " ".join([f"--{k} {v}" for k, v in args.items()])
    # print(f"Запуск команды на {ip}:{ssh_config["port"]} : {cmd}")

    host = ssh_config.pop('hostname')

    async with asyncssh.connect(host, **ssh_config, known_hosts=None) as conn:
        process = await conn.create_process(cmd)

    ssh_config['hostname'] = host


def collect_remote_models(nodes: list[dict], remote_model_dir: str, remote_model_name:str, local_model_dir: str) -> list[str]:
    os.makedirs(local_model_dir, exist_ok=True)
    local_model_paths = []

    for config in nodes:
        remote_model_path = f"{remote_model_dir}/{remote_model_name}"
        local_model_path = os.path.join(local_model_dir, f"model_{config['hostname'].replace('.', '_')}_{config['port']}.pt")
        
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(**config)

        # Скачиваем файл через SFTP
        with ssh.open_sftp() as sftp:
            sftp.get(remote_model_path, local_model_path)
            # print(f"Модель скачана с {ip}:{config["port"]} : {local_model_path}")
        local_model_paths.append(local_model_path)

    return local_model_paths


async def load_nodes_config(config_path="conf.json") -> list[dict]:
    """
    Загружает конфигурацию узлов из файла JSON.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    with open(config_path, "r", encoding="utf-8") as config_file:
        nodes = json.load(config_file)

    # Валидация структуры данных
    if not isinstance(nodes, list):
        raise ValueError("Configuration file must contain a list of nodes.")

    nodes = [json.loads(node) for node in nodes]

    # rename param ip to hostname
    for node in nodes:
        ip = node.pop('ip')
        node['hostname'] = ip
    
    return nodes


async def main():
    dataset_path = "data/mnist.pt"
    output_dir = "splits/"
    n_splits = 2

    nodes = await load_nodes_config()

    remote_dir = "/app/dataset_parts"
    remote_script_path = "/app/train.py"
    remote_model_dir = "/app/models"
    local_model_dir = "local_models/"
    aggregated_model_path = "final_model.pt"

    global_epochs = 4

    # print("Разделение датасета на части...")
    split_files = split_dataset(dataset_path, n_splits, output_dir)

    # print("Распределение частей датасета по узлам...")
    distribute_files_to_nodes(split_files, nodes, remote_dir)

    for global_epoch in range(global_epochs):
        # print(f"Глобальная эпоха {global_epoch + 1}/{global_epochs}")

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

        # print("Сбор моделей с узлов...")
        node_models = collect_remote_models(nodes, remote_model_dir, f"model_{global_epoch}.pt", local_model_dir)

        # print("Агрегация моделей...")
        aggregate_models(node_models, aggregated_model_path, input_dim=28 * 28, output_dim=10)

        # Распространение глобальной модели на узлы
        distribute_global_model(aggregated_model_path, nodes, remote_model_dir)

    # print(f"Финальная модель сохранена в {aggregated_model_path}")



if __name__ == "__main__":
    asyncio.run(main())
