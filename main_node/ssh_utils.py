import os
import paramiko


def distribute_files_to_nodes(files: list, nodes: dict[str, dict], remote_dir: str):
    for i, (ip, config) in enumerate(nodes.items()):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ip, **config)

        # Убедиться, что папка существует
        ssh.exec_command(f"mkdir -p {remote_dir}")

        # Загружаем файл
        file_to_upload = files[i]
        remote_file_path = os.path.join(remote_dir, os.path.basename(file_to_upload))

        with ssh.open_sftp() as sftp:
            sftp.put(file_to_upload, remote_file_path)
            print(f"Файл {file_to_upload} отправлен на {ip}")
