from uuid import UUID

from slet_sdk.core.mixins import BaseResource
from slet_sdk.schemas import SuccessResponse
from slet_sdk.aelite.schemas.chats import (
    GetUserChatsResponse,
    ChatWithoutHistory,
    ChatWithHistory,
)


class ChatsResource(BaseResource):

    async def new_chat(self, title: str | None = None) -> ChatWithoutHistory:
        """Создание нового чата (бизнес модель, а не реальный чат, нужно для сопоставления чата с его владельцем)"""
        return await self._request(
            "POST",
            "/chats/",
            json={"title": title} if title else None,
            schema=ChatWithoutHistory,
        )

    async def delete_chat(self, chat_id: UUID) -> SuccessResponse:
        """Безвозвратное удаление чата вместе со всей историей сообщений."""
        return await self._request(
            "DELETE",
            f"/chats/{chat_id}",
            schema=SuccessResponse,
        )

    async def rename_chat(self, chat_id: UUID, title: str) -> ChatWithoutHistory:
        """Переименование чата. Возвращает обновлённый объект чата."""
        return await self._request(
            "PATCH",
            f"/chats/{chat_id}/rename",
            json={"title": title},
            schema=ChatWithoutHistory,
        )

    async def get_chats(self) -> list[ChatWithoutHistory]:
        """Список всех чатов текущего пользователя, отсортированных по дате создания (новые первые)."""
        response: GetUserChatsResponse = await self._request(
            "GET",
            "/chats/",
            schema=GetUserChatsResponse,
        )
        return response.chats

    async def get_chat(self, chat_id: UUID) -> ChatWithHistory:
        """Получение чата с полной историей сообщений по его UUID."""
        return await self._request(
            "GET",
            f"/chats/{chat_id}",
            schema=ChatWithHistory,
        )
