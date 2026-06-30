from __future__ import annotations
from typing import Literal
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, model_validator

from slet_sdk.aelite.manifest import PermissionLevel, PermissionLevelRestricted


# ---
# PERMISSIONS ENUMS
# ---

Permissions = Literal["connect", "edit", "delete", "read_history"]


class ThreadPermissionEntry(BaseModel):
    user_id: int
    permission: Permissions
    type: Literal["allow", "deny"]


class ThreadPermissions(BaseModel):
    connect: PermissionLevel = PermissionLevel.owner_only
    edit: PermissionLevelRestricted = PermissionLevelRestricted.owner_only
    delete: PermissionLevelRestricted = PermissionLevelRestricted.owner_only
    read_history: PermissionLevel = PermissionLevel.owner_only
    entries: list[ThreadPermissionEntry] = []


# ---
# RESPONSES & INTERNAL USE
# ---


class ThreadWithoutHistory(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID  # обычно UUIDv7
    title: str
    created_at: datetime = Field(description="ISO 8601, RFC 3339")
    permissions: ThreadPermissions


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


class ThreadWithHistory(ThreadWithoutHistory):
    messages: list[MessageSchema]


class ThreadCreateResponse(ThreadWithoutHistory):
    session_ids: dict[str, str] = Field(
        description="Session IDs associated with this agent"
    )


class GetUserThreadsResponse(BaseModel):
    threads: list[ThreadWithoutHistory]


# ---
# REQUESTS
# ---


class ThreadCreateAndUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=128)
    permissions: ThreadPermissions | None = Field(default=None),


class ThreadCreate(ThreadCreateAndUpdate):
    agent_id: UUID


class ThreadUpdate(ThreadCreateAndUpdate):
    @model_validator(mode="after")
    def at_least_one_field(self) -> ThreadUpdate:
        if self.title is None and self.permissions is None:
            raise ValueError("At least one of 'title' or 'permissions' must be provided")
        return self
