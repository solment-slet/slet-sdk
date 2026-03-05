from __future__ import annotations
from typing import TYPE_CHECKING, Any, Type, TypeVar

from pydantic import BaseModel

from slet_sdk.core.typing import LoggerLike

T = TypeVar("T", bound=BaseModel)

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
    async def _request(
        self,
        method: str,
        url: str,
        schema: Type[T] | None = None,
        **kwargs: Any,
    ):
        return await self._client.request(
            method=method,
            url=url,
            schema=schema,
            **kwargs,
        )

    @property
    def _access_token(self) -> str | None:
        return self._client.access_token

    @property
    def _refresh_token(self) -> str | None:
        return self._client.refresh_token

    @property
    def _base_url(self) -> str:
        return self._client.base_url

    @property
    def _base_ws_url(self) -> str:
        return self._client.base_ws_url

    @property
    def logger(self) -> LoggerLike:
        return self._client.logger