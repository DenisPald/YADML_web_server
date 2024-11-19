from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import secrets
from os import environ

app = FastAPI()

# Простые "моковые" данные для авторизации
USERNAME = environ["USERNAME"]
PASSWORD = environ["PASSWORD"]
TOKEN = environ["TOKEN"]

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

# @app.get("/")
# async def protected_route(token: str = Depends(verify_token)):
#     """Защищенный роут, доступный только при наличии корректного токена."""
#     return {"message": "You are authorized"}


# @app.get("/open")
# async def open_route():
#     """Незащищенный роут, доступный всем."""
#     return {"message": "Welcome to the open route!"}

