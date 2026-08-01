from __future__ import annotations
from typing import Literal
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, model_validator

from slet_sdk.aelite.manifest import PermissionLevel, PermissionLevelRestricted, AgentManifest

# ---
# PERMISSIONS ENUMS
# ---

Permissions = Literal["connect", "edit", "delete", "read_info", "read_manifest", "read_history"]


class ThreadPermissionEntry(BaseModel):
    user_id: int
    permission: Permissions
    type: Literal["allow", "deny"]


class ThreadPermissions(BaseModel):
    connect: PermissionLevel = PermissionLevel.owner_only
    edit: PermissionLevelRestricted = PermissionLevelRestricted.owner_only
    delete: PermissionLevelRestricted = PermissionLevelRestricted.owner_only
    read_info: PermissionLevel = PermissionLevel.owner_only
    read_manifest: PermissionLevel = PermissionLevel.owner_only
    read_history: PermissionLevel = PermissionLevel.owner_only
    entries: list[ThreadPermissionEntry] = []


# ---
# RESPONSES & INTERNAL USE
# ---


class ThreadInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    permissions: ThreadPermissions
    source_agent_id: UUID | None = Field(default=None)
    source_agent_version: str | None = Field(default=None)
    manifest_version: str | None = Field(default=None)


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


class ThreadWithHistory(ThreadInfo):
    messages: list[MessageSchema]


class ThreadWithManifest(ThreadInfo):
    manifest: AgentManifest | None = None


class ThreadCreateResponse(ThreadInfo):
    session_ids: dict[str, str] = Field(
        description="Session IDs associated with this agent"
    )


class GetUserThreadsResponse(BaseModel):
    threads: list[ThreadInfo]


# ---
# REQUESTS
# ---


class ThreadCreateAndUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=128)
    manifest: AgentManifest | None = Field(default=None)
    permissions: ThreadPermissions | None = Field(default=None)


class ThreadCreate(ThreadCreateAndUpdate):
    agent_id: UUID | None = None

    @model_validator(mode="after")
    def check_manifest_source(self):
        if self.agent_id is None and self.manifest is None:
            raise ValueError("Either agent_id or manifest must be provided")
        if self.agent_id is not None and self.manifest is not None:
            raise ValueError("Provide either agent_id or manifest, not both")
        return self


class ThreadUpdate(ThreadCreateAndUpdate):
    agent_id: UUID | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> ThreadUpdate:
        if self.agent_id is None and self.title is None and self.manifest is None and self.permissions is None:
            raise ValueError("At least one of 'agent', 'title', 'manifest' or 'permissions' must be provided")
        return self

    @model_validator(mode="after")
    def check_manifest_source(self):
        if self.agent_id is not None and self.manifest is not None:
            raise ValueError("Provide either agent_id or manifest, not both")
        return self
