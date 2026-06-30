from uuid import UUID

from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.schemas.agents import (
    AgentInfo,
    AgentInfoWithManifest,
    GetUserAgentsResponse,
)


class AgentsResource(BaseResource):

    async def create_agent(self, agent_manifest: AgentManifest) -> AgentInfo:
        """Create new agent."""
        return await self._request(
            "POST",
            "/agents/",
            schema=AgentInfo,
            body=agent_manifest,
        )

    async def update_agent(self, agent_id: UUID, agent_manifest: AgentManifest) -> AgentInfo:
        """Updating an existing agent."""
        return await self._request(
            "PUT",
            f"/agents/{agent_id}",
            body=agent_manifest,
            schema=AgentInfo,
        )

    async def delete_agent(self, agent_id: UUID) -> None:
        """Complete removal of the agent along with all threads created on its basis."""
        return await self._request(
            "DELETE",
            f"/agents/{agent_id}",
        )

    async def list_agents(self) -> list[AgentInfo]:
        """The list of agents owned by the user."""
        response = await self._request(
            "GET",
            "/agents/",
            schema=GetUserAgentsResponse,
        )
        return response.agents

    async def get_agent(self, agent_id: UUID) -> AgentInfoWithManifest:
        """Receiving an agent with his manifest."""
        return await self._request(
            "GET",
            f"/agents/{agent_id}",
            schema=AgentInfoWithManifest,
        )
