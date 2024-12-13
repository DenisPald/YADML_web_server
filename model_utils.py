import torch
from torch import nn
from model import LeNet
import asyncssh

async def distribute_global_model(global_model_path: str, nodes: list[dict], remote_model_dir: str):
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

def aggregate_models(model_paths: list, output_path: str, input_dim: int, output_dim: int):
    aggregated_model = LeNet()
    aggregated_state_dict = None
    for path in model_paths:
        model_state_dict = torch.load(path)
        if aggregated_state_dict is None:
            aggregated_state_dict = model_state_dict
        else:
            for key in aggregated_state_dict:
                aggregated_state_dict[key] += model_state_dict[key]

    if aggregated_state_dict is not None:
        for key in aggregated_state_dict:
            aggregated_state_dict[key] /= len(model_paths)

        aggregated_model.load_state_dict(aggregated_state_dict)
        torch.save(aggregated_model.state_dict(), output_path)

