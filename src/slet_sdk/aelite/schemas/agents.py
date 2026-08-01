from typing import Literal
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from slet_sdk.aelite.manifest import AgentPermissions, AgentManifest

Permissions = Literal["view_manifest", "create_thread", "deploy", "delete"]


class AgentInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version: str
    created_at: datetime
    updated_at: datetime
    permissions: AgentPermissions


class AgentInfoWithManifest(AgentInfo):
    manifest: AgentManifest


class GetUserAgentsResponse(BaseModel):
    agents: list[AgentInfo]
