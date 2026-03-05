from slet_sdk.core.mixins import BaseResource

class ChatsResource(BaseResource):
    async def create(self, title: str | None = None):
        return await self._request("POST", "/ae/chats/", json={"title": title} if title else None)