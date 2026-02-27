import re
from pydantic import BaseModel, EmailStr, ConfigDict, field_validator
from slet_sdk.core.schemas import SuccessResponse


class UserRefresh(BaseModel):
    refresh_token: str


class UserRefreshResponse(SuccessResponse):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserLogin(BaseModel):
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str):
        return v.lower().strip()


class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str):
        return v.lower().strip()

    @field_validator("password")
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Пароль должен содержать минимум 8 символов")
        if len(v) > 200:
            raise ValueError("Пароль не может содержать более 200 символов")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну заглавную букву")
        if not re.search(r"[a-z]", v):
            raise ValueError("Пароль должен содержать хотя бы одну строчную букву")
        if not re.search(r"[0-9]", v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        return v  # обязательно возвращаем значение


class UserInfo(BaseModel):
    """
    Информация о пользователе
    """

    id: int
    name: str
    email: str

    model_config = ConfigDict(from_attributes=True)


class UserAuthResponse(UserRefreshResponse):
    """
    Ответ на авторизацию (вход/регистрация)
    """

    user: UserInfo


class UserLoginResponse(UserAuthResponse):
    """
    Ответ на авторизацию (вход)
    """


class UserRegisterResponse(UserAuthResponse):
    """
    Ответ на регистрацию (регистрация)
    """

    status: int = 201
