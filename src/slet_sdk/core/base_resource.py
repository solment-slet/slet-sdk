from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slet_sdk.slet_client import SletClient


class BaseResource:
    """
    Базовый класс для всех namespace-ресурсов.
    Каждый ресурс получает ссылку на родительский клиент
    и использует его для HTTP-запросов.
    """

    def __init__(self, client: SletClient) -> None:
        self._client = client

    # Удобные шорткаты к методам клиента
    async def _request(self, method: str, url: str, **kwargs):
        return await self._client.request(method, url, **kwargs)

    @property
    def _access_token(self) -> str | None:
        return self._client.access_token

    @property
    def _refresh_token(self) -> str | None:
        return self._client.refresh_token

    @property
    def _base_url(self) -> str:
        return self._client.base_url