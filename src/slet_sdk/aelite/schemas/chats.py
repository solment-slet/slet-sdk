from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field
from slet_sdk.core.schemas import SuccessResponse


class ChatSchema(BaseModel):
    """Pydantic схема для чата (не SQLAlchemy модель!)"""

    id: UUID # обычно UUIDv7
    title: str
    created_at: datetime = Field(description="ISO 8601, RFC 3339")

    class Config:
        from_attributes = True  # для конвертации SQLAlchemy -> Pydantic


class ChatCreate(BaseModel):
    title: str | None = Field(default=None, max_length=128)


class ChatCreateResponse(ChatSchema, SuccessResponse):
    status: int = 201


class GetUserChatsResponse(SuccessResponse):
    chats: list[ChatSchema]
