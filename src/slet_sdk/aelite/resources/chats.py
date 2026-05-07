from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.schemas.chats import ChatCreateResponse

class ChatsResource(BaseResource):
    async def new_chat(self, title: str | None = None) -> ChatCreateResponse:
        """Создание нового чата (бизнес модель, а не реальный чат, нужно для сопоставления чата с его владельцем)"""
        return await self._request(
            "POST",
            "/ae/chats/",
            json={"title": title} if title else None,
            schema=ChatCreateResponse,
        )