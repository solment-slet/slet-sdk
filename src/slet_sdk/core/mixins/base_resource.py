from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from slet_sdk.slet_client import SletClient
    from slet_sdk.typing import LoggerLike, WebsocketsModule

T = TypeVar("T", bound=BaseModel)


class BaseResource:
    """
    Базовый класс для всех namespace-ресурсов.
    Каждый ресурс получает ссылку на родительский клиент
    и использует его для HTTP-запросов.
    """

    def __init__(self, client: SletClient, route_prefix: str) -> None:
        self._client = client
        self._route_prefix = route_prefix

    async def _request(
        self,
        method: str,
        url: str,
        schema: type[T] | None = None,
        **kwargs: Any,
    ) -> dict | T:
        return await self._client.request(
            method=method,
            url=self._route_prefix + url,
            schema=schema,
            **kwargs,
        )

    @property
    def logger(self) -> LoggerLike:
        return self._client.logger

    @property
    def websockets(self) -> WebsocketsModule:
        return self._client.websockets

    @property
    def base_url(self) -> str:
        return self._client.base_url + "/" + self._route_prefix

    @property
    def base_ws_url(self) -> str:
        return self._client.base_ws_url + "/" + self._route_prefix

    @property
    def _access_token(self) -> str | None:
        return self._client.access_token

    @property
    def _refresh_token(self) -> str | None:
        return self._client.refresh_token
