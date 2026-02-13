from typing import Any
from pydantic import BaseModel, Field


# Базовые модели для успеха и ошибок
class SuccessResponse(BaseModel):
    """
    Базовая модель для успешных ответов. (не обязательно используется во всех ручках)
    """

    status: int = 200


class ErrorResponse(BaseModel):
    """
    Базовая модель для ошибок. (не обязательно используется во всех ручках)
    """

    status: int
    error: str
    message: str
    extra: dict[str, Any | None] = Field(
        default_factory=dict
    )  # Если extra нет, то будет {}
