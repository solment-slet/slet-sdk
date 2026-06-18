import re

from pydantic import BaseModel, EmailStr, ConfigDict, field_validator
from slet_sdk.schemas import SuccessResponse


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
            raise ValueError("Password must contain at least 8 characters")
        if len(v) > 200:
            raise ValueError("Password cannot contain more than 200 characters")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserInfo(BaseModel):
    """
    Информация о пользователе
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    email: str


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
