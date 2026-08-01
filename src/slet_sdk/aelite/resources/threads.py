from uuid import UUID
from typing import overload

from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.aelite.schemas.agents import AgentInfo
from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.schemas.threads import (
    GetUserThreadsResponse,
    ThreadInfo,
    ThreadWithHistory,
    ThreadWithManifest,
    ThreadPermissions,
    ThreadCreate,
    ThreadUpdate,
    ThreadCreateResponse,
)


class ThreadsResource(BaseResource):

    @overload
    async def create_thread(
        self,
        manifest: AgentManifest,
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
        agent: UUID | AgentInfo | None = None,
    ) -> ThreadCreateResponse: ...

    @overload
    async def create_thread(
        self,
        agent: UUID | AgentInfo,
        title: str | None = None,
        manifest: AgentManifest | None = None,
        permissions: ThreadPermissions | None = None,
    ) -> ThreadCreateResponse: ...

    @overload
    async def create_thread(
        self, *, payload: ThreadCreate,
    ) -> ThreadCreateResponse: ...

    async def create_thread(
        self,
        *,
        title: str | None = None,
        manifest: AgentManifest | None = None,
        permissions: ThreadPermissions | None = None,
        agent: UUID | AgentInfo | None = None,
        payload: ThreadCreate | None = None,
    ) -> ThreadCreateResponse:
        """Create a new thread."""
        if payload is not None:
            return await self._request(
                "POST",
                "/threads/",
                body=payload,
                schema=ThreadCreateResponse,
            )
        if agent is not None or manifest is not None:
            if agent is not None:
                if isinstance(agent, UUID):
                    agent_id = agent
                elif isinstance(agent, AgentInfo):
                    agent_id = agent.id
                else:
                    raise TypeError("Argument 'agent' must be a UUID or AgentInfo.")
            else:
                agent_id = None

            return await self._request(
                "POST",
                "/threads/",
                json={
                    "title": title,
                    "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
                    "permissions": permissions.model_dump(mode="json") if permissions is not None else None,
                    "agent_id": agent_id,
                },
                schema=ThreadCreateResponse,
            )

        raise ValueError("Either 'agent', 'manifest' or 'payload' must be provided.")

    @overload
    async def update_thread(
        self,
        thread_id: UUID,
        *,
        agent_id: UUID | None = None,
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
    ) -> ThreadInfo: ...

    @overload
    async def update_thread(
        self, thread_id: UUID, *, payload: ThreadUpdate,
    ) -> ThreadInfo: ...

    async def update_thread(
        self,
        thread_id: UUID,
        *,
        title: str | None = None,
        manifest: AgentManifest | None = None,
        permissions: ThreadPermissions | None = None,
        agent_id: UUID | None = None,
        payload: ThreadUpdate | None = None,
    ) -> ThreadInfo:
        """Updating an existing thread."""
        if payload is not None:
            return await self._request(
                "PUT",
                f"/threads/{thread_id}",
                body=payload,
                schema=ThreadInfo,
            )
        if agent_id is not None or title is not None or permissions is not None:
            return await self._request(
                "PUT",
                f"/threads/{thread_id}",
                json={
                    "title": title,
                    "manifest": manifest.model_dump(mode="json") if manifest is not None else None,
                    "permissions": permissions.model_dump(mode="json") if permissions is not None else None,
                    "agent_id": agent_id,
                },
                schema=ThreadInfo,
            )

        raise ValueError("Either 'agent', 'title', 'permissions' or 'payload' must be provided.")

    async def delete_thread(self, thread_id: UUID) -> None:
        """Permanent deletion of the thread along with the entire message history."""
        return await self._request(
            "DELETE",
            f"/threads/{thread_id}",
        )

    async def list_threads(self) -> list[ThreadInfo]:
        """List of current user's threads."""
        response: GetUserThreadsResponse = await self._request(
            "GET",
            "/threads/",
            schema=GetUserThreadsResponse,
        )
        return response.threads

    async def get_thread(self, thread_id: UUID) -> ThreadWithManifest:
        """Getting a thread with a manifest."""
        return await self._request(
            "GET",
            f"/threads/{thread_id}",
            schema=ThreadWithManifest,
        )

    async def get_thread_with_history(self, thread_id: UUID) -> ThreadWithHistory:
        """Getting a thread with a full message history."""
        return await self._request(
            "GET",
            f"/threads/{thread_id}/history",
            schema=ThreadWithHistory,
        )
