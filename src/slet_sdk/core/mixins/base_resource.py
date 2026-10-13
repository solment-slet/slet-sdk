from __future__ import annotations
from typing import Any, TypeVar, TYPE_CHECKING

from pydantic import BaseModel
from websockets.exceptions import InvalidStatus

if TYPE_CHECKING:
    from websockets.asyncio.connection import Connection
    from slet_sdk.slet_client import SletClient
    from slet_sdk.types import LoggerLike, WebsocketsModule

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
        *,
        body: BaseModel | dict | None = None,
        **kwargs: Any,
    ) -> dict | T | None:
        return await self._client.request(
            method=method,
            url=self._route_prefix + url,
            schema=schema,
            body=body,
            **kwargs,
        )

    async def _get_access_token(self) -> str:
        """Актуальный access-токен (обновляется только если он истёк)."""
        return await self._client.get_access_token()

    async def _auth_headers(self) -> dict[str, str]:
        """Заголовок Authorization с актуальным токеном."""
        return {"Authorization": f"Bearer {await self._get_access_token()}"}

    async def _connect_ws(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Connection:
        """
        Открывает WebSocket-соединение с авторизацией.

        Токен проверяется локально по `exp` перед подключением. Если сервер
        всё же отклонил handshake (401/403: расхождение часов, токен отозван),
        токены обновляются и подключение повторяется один раз.

        Args:
            url (str): Полный ws(s)-URL.
            headers (dict[str, str] | None): Дополнительные заголовки.
                Authorization перезаписывается актуальным токеном.
            **kwargs: Остальные аргументы `websockets.connect`.

        Returns:
            Connection: Открытое соединение. Можно использовать как
            `async with await self._connect_ws(url) as ws`.

        Raises:
            InvalidStatus: Если сервер отклонил handshake после повтора
                или статус не 401/403.
            SletClientError: Если не удалось обновить токены.
        """
        for attempt in range(2):
            token = await self._get_access_token()
            try:
                # noinspection PyTypeChecker
                return await self.websockets.connect(
                    url,
                    additional_headers={
                        **(headers or {}),
                        "Authorization": f"Bearer {token}",
                    },
                    **kwargs,
                )
            except InvalidStatus as e:
                if (
                    attempt == 0
                    and e.response.status_code in (401, 403)
                    and self._client.refresh_token
                ):
                    await self._client.refresh_if_stale(token)
                    continue
                raise
        raise RuntimeError("unreachable")

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
