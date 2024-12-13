import os
import asyncssh

async def distribute_files_to_nodes(files: list, nodes: list[dict], remote_dir: str):
    for i, config in enumerate(nodes):
        host = config['hostname']
        port = config['port']
        username = config.get('username', 'root')
        password = config.get('password', None)
        file_to_upload = files[i]
        remote_file_path = os.path.join(remote_dir, os.path.basename(file_to_upload))

        try:
            async with asyncssh.connect(
                host=host,
                port=port,
                username=username,
                password=password,
                known_hosts=None
            ) as conn:
                await conn.run(f"mkdir -p {remote_dir}", check=True)
                sftp = await conn.start_sftp_client()
                await sftp.put(file_to_upload, remote_file_path)
                print(f"Uploaded {file_to_upload} to {host}:{remote_file_path}")
        except Exception as e:
            print(f"Failed to distribute file to {host}: {e}")
            # Можно добавить логику по удалению узлов или обработке ошибок

async def collect_remote_models(nodes: list[dict], remote_model_dir: str, remote_model_name: str, local_model_dir: str) -> list[str]:
    os.makedirs(local_model_dir, exist_ok=True)
    local_model_paths = []

    bad_nodes = []
    for config in nodes:
        host = config['hostname']
        port = config['port']
        username = config.get('username', 'root')
        password = config.get('password', None)

        remote_model_path = f"{remote_model_dir}/{remote_model_name}"
        local_model_path = os.path.join(local_model_dir, f"model_{host.replace('.', '_')}_{port}.pt")

        try:
            async with asyncssh.connect(
                host=host,
                port=port,
                username=username,
                password=password,
                known_hosts=None
            ) as conn:
                sftp = await conn.start_sftp_client()
                print(f"Checking files in {remote_model_dir} on node {host}")
                files = await sftp.listdir(remote_model_dir)
                print(files)  # Вывод списка файлов для отладки

                if remote_model_name not in files:
                    print(f"File {remote_model_path} not found on node {host}")
                    continue

                await sftp.get(remote_model_path, local_model_path)
                print(f"Downloaded {remote_model_path} to {local_model_path}")
                local_model_paths.append(local_model_path)
        except Exception as e:
            print(f"Failed to connect or download from {host}: {e}")
            bad_nodes.append(config)

    for bn in bad_nodes:
        if bn in nodes:
            nodes.remove(bn)

    return local_model_paths

