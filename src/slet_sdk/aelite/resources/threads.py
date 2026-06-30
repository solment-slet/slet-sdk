from uuid import UUID

from slet_sdk.core.mixins import BaseResource
from slet_sdk.schemas import SuccessResponse
from slet_sdk.aelite.schemas.threads import (
    GetUserThreadsResponse,
    ThreadWithoutHistory,
    ThreadWithHistory,
)


class ThreadsResource(BaseResource):

    async def create_thread(self, title: str | None = None) -> ThreadWithoutHistory:
        """Создание нового треда"""
        return await self._request(
            "POST",
            "/threads/",
            json={"title": title} if title else None,
            schema=ThreadWithoutHistory,
        )

    async def delete_thread(self, thread_id: UUID) -> SuccessResponse:
        """Безвозвратное удаление треда вместе со всей историей сообщений."""
        return await self._request(
            "DELETE",
            f"/threads/{thread_id}",
            schema=SuccessResponse,
        )

    async def update_thread(self, thread_id: UUID, title: str) -> ThreadWithoutHistory:
        """Переименование треда."""
        return await self._request(
            "PATCH",
            f"/threads/{thread_id}/rename",
            json={"title": title},
            schema=ThreadWithoutHistory,
        )

    async def list_threads(self) -> list[ThreadWithoutHistory]:
        """Список тредов текущего пользователя, отсортированных по дате создания (новые первые)."""
        response: GetUserThreadsResponse = await self._request(
            "GET",
            "/threads/",
            schema=GetUserThreadsResponse,
        )
        return response.threads

    async def get_thread(self, thread_id: UUID) -> ThreadWithHistory:
        """Получение треда с полной историей сообщений по его UUID."""
        return await self._request(
            "GET",
            f"/threads/{thread_id}",
            schema=ThreadWithHistory,
        )
