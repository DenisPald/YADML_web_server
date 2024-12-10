from fastapi import FastAPI, Depends, HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
import secrets
import os
import json
import subprocess

app = FastAPI()

USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]
TOKEN = os.environ["TOKEN"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")

PROCESS_TRACKER = {"running": False}

class AuthResponse(BaseModel):
    token: str

def verify_credentials(username: str, password: str) -> bool:
    """Проверка имени пользователя и пароля."""
    return secrets.compare_digest(username, USERNAME) and secrets.compare_digest(password, PASSWORD)

@app.post("/auth", response_model=AuthResponse)
async def auth(form_data: OAuth2PasswordRequestForm = Depends()):
    """Роут для аутентификации. Принимает логин и пароль, возвращает токен при успешной аутентификации."""
    if not verify_credentials(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthResponse(token=TOKEN)

def verify_token(token: str = Depends(oauth2_scheme)):
    """Проверка валидности токена."""
    if token != TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

@app.post("/upload")
async def upload_file(
    file: UploadFile,
    token: str = Depends(verify_token)
):
    """Прием датасета от авторизованного пользователя и сохранение его в папку."""
    # Создаем папку для датасета
    os.makedirs('data/', exist_ok=True)
    
    # Полный путь для сохранения файла
    file_path = 'data/mnist.pt'
    content = await file.read()

    # Сохранение файла
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content.decode('utf-8'))

    return {"message": "OK"}


class WorkerConfig(BaseModel):
    ip: str
    port: str
    username: str
    password: str


@app.post("/run")
async def run_training(
    workers: list[WorkerConfig], token: str = Depends(verify_token)
):
    """
    Принимает список воркеров, сохраняет их в конфигурационный файл и запускает скрипт в отдельном процессе.
    """
    # Проверка, что процесс не запущен
    if PROCESS_TRACKER["running"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training process is already running."
        )
    
    # Сохраняем конфигурацию в `conf.json`
    config_path = "main_node/conf.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump([worker.model_dump_json() for worker in workers], config_file, ensure_ascii=False, indent=4)

    # Запускаем скрипт в отдельном процессе
    process = subprocess.Popen(["python3", "main_node/run.py"])
    PROCESS_TRACKER["running"] = True

    def monitor_process():
        process.wait()  
        PROCESS_TRACKER["running"] = False

    # Запускаем мониторинг в фоне
    import threading
    threading.Thread(target=monitor_process, daemon=True).start()

    return {"message": "OK"}


@app.get("/get")
async def get_final_model(token: str = Depends(verify_token)):
    """
    Проверяет завершение процесса обучения и возвращает файл модели, если он существует.
    """
    model_path = "final_model.pt"
    if PROCESS_TRACKER["running"]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model is still learning"
        )
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Final model is not found"
        )

    # Возвращаем файл модели
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename="final_model.pt"
    )


