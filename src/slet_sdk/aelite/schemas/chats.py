from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

# ---
# RESPONSES & INTERNAL USE
# ---


class ChatWithoutHistory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID  # обычно UUIDv7
    title: str
    created_at: datetime = Field(description="ISO 8601, RFC 3339")


class ToolCallSchema(BaseModel):
    id: str
    name: str
    args: dict


class MessageSchema(BaseModel):
    id: str
    role: str  # 'user', 'ai', 'tool'
    content: str
    created_at: datetime

    # Для AI сообщений (если были вызовы инструментов)
    tool_calls: list[ToolCallSchema] = []

    # Для Tool сообщений (информация о результате)
    tool_call_id: str | None = None
    tool_name: str | None = None


class ChatWithHistory(ChatWithoutHistory):
    messages: list[MessageSchema]


# ---
# REQUESTS
# ---


class ChatCreateAndRename(BaseModel):
    title: str | None = Field(default=None, max_length=128)


class ChatCreate(ChatCreateAndRename):
    pass


class ChatRename(ChatCreateAndRename):
    pass


# ---
# RESPONSES
# ---


class GetUserChatsResponse(BaseModel):
    chats: list[ChatWithoutHistory]
