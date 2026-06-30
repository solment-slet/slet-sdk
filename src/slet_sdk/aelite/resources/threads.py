from uuid import UUID

from slet_sdk.core.mixins import BaseResource
from slet_sdk.schemas import SuccessResponse
from slet_sdk.aelite.schemas.threads import (
    GetUserThreadsResponse,
    ThreadInfo,
    ThreadWithHistory,
    ThreadPermissions,
    ThreadCreate, ThreadUpdate,
)


class ThreadsResource(BaseResource):

    async def create_thread(
        self,
        *,
        agent_id: UUID | None = None,
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
        thread_schema: ThreadCreate | None = None,
    ) -> ThreadInfo:
        """Create a new thread."""
        if thread_schema is not None:
            return await self._request(
                "POST",
                "/threads/",
                body=thread_schema,
                schema=ThreadInfo,
            )
        if agent_id is not None:
            return await self._request(
                "POST",
                "/threads/",
                json={
                    "agent_id": agent_id,
                    "title": title,
                    "permissions": permissions.model_dump_json() if permissions is not None else None,
                },
                schema=ThreadInfo,
            )

        raise ValueError("Either 'agent_id' or 'thread_schema' must be provided.")

    async def update_thread(
        self,
        *,
        agent_id: UUID | None = None,
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
        thread_schema: ThreadUpdate | None = None,
    ) -> ThreadInfo:
        """Updating an existing thread."""
        if thread_schema is not None:
            return await self._request(
                "PUT",
                "/threads/",
                body=thread_schema,
                schema=ThreadInfo,
            )
        if agent_id is not None or title is not None or permissions is not None:
            return await self._request(
                "PUT",
                "/threads/",
                json={
                    "agent_id": agent_id,
                    "title": title,
                    "permissions": permissions.model_dump_json() if permissions is not None else None,
                },
                schema=ThreadInfo,
            )

        raise ValueError("Either 'agent_id', 'title', 'permissions' or 'thread_schema' must be provided.")

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

    async def get_thread(self, thread_id: UUID) -> ThreadWithHistory:
        """Getting a thread with a full message history."""
        return await self._request(
            "GET",
            f"/threads/{thread_id}",
            schema=ThreadWithHistory,
        )
