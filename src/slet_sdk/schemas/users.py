from pydantic import BaseModel, EmailStr, IPvAnyAddress
from slet_sdk.typing import UserPrivilege


class UserContext(BaseModel):
    """Легковесная модель из JWT - без похода в БД."""

    id: int
    name: str
    email: EmailStr
    privilege: UserPrivilege


class MyIPResponse(BaseModel):
    ip: IPvAnyAddress
