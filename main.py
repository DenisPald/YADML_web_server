from fastapi import FastAPI, Depends, HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from pydantic import BaseModel
import secrets
from dotenv import load_dotenv
import os
import json
import subprocess

app = FastAPI()

load_dotenv("./env_file.txt")
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

class AuthJsonModel(BaseModel):
    username: str
    password: str

@app.post("/auth", response_model=AuthResponse)
async def auth(auth_json: AuthJsonModel):
    """Роут для аутентификации. Принимает логин и пароль, возвращает токен при успешной аутентификации."""
    if not verify_credentials(auth_json.username, auth_json.password):
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
    with open(file_path, 'wb') as f:
        f.write(content)

    return {"message": "OK"}


@app.post("/run")
async def run_training(
    workers: dict, token: str = Depends(verify_token)
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
    
    config_path = "conf.json"
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(workers, config_file, ensure_ascii=False, indent=4)

    # Запускаем скрипт в отдельном процессе
    process = subprocess.Popen(
        ["python3", "run.py"],
    )
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
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Final model is not found"
        )

    # Возвращаем файл модели
    return FileResponse(
        model_path,
        media_type="application/octet-stream",
        filename="final_model.pt"
    )


