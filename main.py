from fastapi import FastAPI, Depends, HTTPException, status, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import secrets
import os

app = FastAPI()

USERNAME = os.environ["USERNAME"]
PASSWORD = os.environ["PASSWORD"]
TOKEN = os.environ["TOKEN"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")

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
    file_path = 'data/dataset.csv'
    content = await file.read()

    # Сохранение файла
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content.decode('utf-8'))

    return {"message": "OK"}

