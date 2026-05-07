from typing import Literal

from pydantic import BaseModel, EmailStr

from slet_sdk.typing import UserPrivilege

class UserContext(BaseModel):
    """Легковесная модель из JWT — без похода в БД."""
    id: int
    name: str
    email: EmailStr
    privilege: UserPrivilege