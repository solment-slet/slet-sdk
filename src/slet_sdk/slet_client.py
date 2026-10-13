from __future__ import annotations

import httpx
import json
import time
import base64
import asyncio
import logging
from typing import Any, TypeVar
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID
from enum import Enum

from pydantic import BaseModel, ValidationError
import websockets

from slet_sdk.schemas import ErrorCode, ErrorResponse
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas.errors import NetworkError
from slet_sdk.schemas.auth import (
    UserLoginResponse,
    UserRegisterResponse,
    UserRefreshResponse,
)
from slet_sdk.aelite.resources.resource import AeliteResource
from slet_sdk.types import LoggerLike, WebsocketsModule
from slet_sdk.config import ApiPrefixes


T = TypeVar('T', bound=BaseModel)


class SletClient:
    """
    Единый асинхронный клиент для всех сервисов Slet API.

    Использование:
        async with SletClient() as client:
            await client.signin("user@example.com", "password")
            await client.aelite.deploy(manifest)
    """

    def __init__(
        self,
        base_url: str,
        *,
        # Network
        timeout: float = 500.0,
        ssl_verify: bool = True,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        websockets_module: WebsocketsModule = websockets,
        # API
        api_versions: ApiPrefixes = ApiPrefixes(),
        # Logging
        logger: LoggerLike = logging.getLogger(__name__),
    ):
        self.logger = logger
        self.base_url = base_url[:-1] if base_url.endswith("/") else base_url
        self.base_ws_url = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        self._client = (
            client
            if client
            else httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(timeout),
                verify=ssl_verify,
                transport=transport,
            )
        )
        self._refresh_lock = asyncio.Lock()
        self.token_leeway: float = 30.0  # обновляем за 30 сек до истечения
        self.websockets = websockets_module
        self.api_versions = api_versions
        self.access_token: str | None = None
        self.refresh_token: str | None = None

        # ── Namespace-ресурсы (продукты) ──────────────────────
        self.aelite = AeliteResource(self, self.api_versions.aelite)

    # ------------------ Context Manager ------------------

    async def __aenter__(self) -> SletClient:
        # Просто возвращаем себя
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Закрываем внутренний httpx клиент
        await self.close()

    # ------------------ Internal Methods ------------------

    @staticmethod
    def _jwt_exp(token: str) -> float | None:
        """Достаёт exp из JWT без проверки подписи."""
        try:
            payload = token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            return json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        except Exception:
            return None

    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            raise SletClientError(
                ErrorResponse(
                    status=0,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Access token is missing. Call signin() first.",
                )
            )
        return {"Authorization": f"Bearer {self.access_token}"}

    async def request(
        self,
        method: str,
        url: str,
        schema: type[T] | None = None,
        *,
        body: BaseModel | dict | None = None,
        **kwargs: Any,
    ) -> dict | T | None:
        return await self._request_impl(method, url, schema=schema, body=body, **kwargs)

    async def _request_impl(
        self,
        method: str,
        url: str,
        schema: type[T] | None = None,
        *,
        body: BaseModel | dict | None = None,
        _skip_refresh: bool = False,
        **kwargs: Any,
    ) -> dict | T | None:
        """
        Выполняет HTTP-запрос с авторизацией, автоматическим обновлением
        токенов и валидацией ответа.

        Токен обновляется в двух случаях:
        1. Проактивно: перед отправкой, если срок жизни access-токена
           (поле `exp` в JWT) истёк или истекает в пределах `token_leeway`.
        2. Реактивно: если сервер ответил 401 с кодом INVALID_ACCESS_TOKEN
           (например, часы клиента и сервера расходятся). В этом случае
           запрос повторяется один раз.

        Args:
            method (str): HTTP-метод.
            url (str): Путь относительно base_url.
            schema (type[T] | None): Pydantic-схема для валидации ответа.
                None означает, что ответ возвращается как есть (dict).
            body (BaseModel | dict | None): Тело запроса. Взаимоисключающе
                с `json` в kwargs.
            _skip_refresh (bool): Служебный флаг. Отключает и проактивный, и
                реактивный refresh. Нужен для самого refresh-запроса (иначе
                рекурсия и deadlock на lock'е) и для signin/signup (там
                токена ещё нет, а старый может быть протухшим).
            **kwargs: Остальные аргументы `httpx.AsyncClient.request`.

        Returns:
            dict | T | None: Провалидированный ответ, сырой dict или None
            для пустого тела.

        Raises:
            ValueError: Если переданы одновременно `body` и `json`.
            SletClientError: При сетевой ошибке, не-2xx ответе, пустом теле
                при заданной схеме, невалидном JSON или несоответствии схеме.
        """
        last_resp: httpx.Response | None = None

        if body is not None and "json" in kwargs:
            raise ValueError("Pass either 'body' or 'json', not both.")

        # ── Сериализация тела ────────────────────────────────
        if body is not None:
            if isinstance(body, BaseModel):
                kwargs["content"] = body.model_dump_json(exclude_none=True)
            else:
                kwargs["content"] = json.dumps(
                    body, default=self._json_default
                ).encode("utf-8")
            kwargs["headers"] = {
                **kwargs.get("headers", {}),
                "Content-Type": "application/json",
            }

        elif "json" in kwargs:
            raw_json = kwargs.pop("json")
            kwargs["content"] = json.dumps(
                raw_json, default=self._json_default
            ).encode("utf-8")
            kwargs["headers"] = {
                **kwargs.get("headers", {}),
                "Content-Type": "application/json",
            }

        # ── Отправка с одним повтором после refresh ──────────
        for attempt in range(2):
            # Проактивный refresh: не ждём 401, если по `exp` токен уже
            # (почти) истёк. Проверка локальная, без запросов к серверу.
            if (
                not _skip_refresh
                and self.access_token
                and self.refresh_token
                and self._token_expired()
            ):
                await self.refresh_if_stale(self.access_token)

            # Запоминаем токен, с которым уходит запрос: если придёт 401,
            # refresh_if_stale поймёт, обновил ли его кто-то параллельно.
            sent_token = self.access_token

            merged_headers = {**kwargs.get("headers", {})}
            if sent_token:
                merged_headers["Authorization"] = f"Bearer {sent_token}"
            request_kwargs = {**kwargs, "headers": merged_headers}

            try:
                resp = await self._client.request(method, url, **request_kwargs)
            except httpx.TransportError as e:
                raise SletClientError(NetworkError(message=str(e)))

            last_resp = resp

            # ── Успешный ответ ───────────────────────────────
            if 200 <= resp.status_code < 300:
                if self._is_empty_body(method, resp):
                    if schema:
                        raise SletClientError(
                            ErrorResponse(
                                status=resp.status_code,
                                error=ErrorCode.VALIDATION_ERROR,
                                message=(
                                    "Expected response body for schema "
                                    "validation, got empty body"
                                ),
                            )
                        )
                    return None

                data = self._parse_json(resp)
                return (
                    self._validate_schema(data, schema, resp.status_code)
                    if schema
                    else data
                )

            # ── Реактивный refresh ───────────────────────────
            # Страховка на случай, если проактивная проверка не сработала
            # (нечитаемый `exp`, расхождение часов, токен отозван).
            if (
                resp.status_code == 401
                and not _skip_refresh
                and attempt == 0
                and self.refresh_token
            ):
                err_data = self._parse_json(resp, allow_fail=True)
                if err_data.get("error") == ErrorCode.INVALID_ACCESS_TOKEN:
                    await self.refresh_if_stale(sent_token)
                    continue

            raise SletClientError(self._build_error(resp))

        assert last_resp is not None
        raise SletClientError(self._build_error(last_resp))

    @staticmethod
    def _json_default(obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    @staticmethod
    def _is_empty_body(method: str, resp: httpx.Response) -> bool:
        if method.upper() == "HEAD":
            return True
        if resp.status_code in (204, 304):
            return True
        if not resp.content:
            return True
        return False

    def _parse_json(self, resp: httpx.Response, allow_fail: bool = False) -> dict:
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            if allow_fail:
                return {}
            raise SletClientError(
                ErrorResponse(
                    status=resp.status_code,
                    error=ErrorCode.VALIDATION_ERROR,
                    message=f"Invalid JSON response: {resp.text}",
                    extra={"exception": str(exc)},
                )
            )

    def _validate_schema(
        self,
        data: dict,
        schema: type[T],
        status: int,
    ) -> T:
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise SletClientError(
                ErrorResponse(
                    status=status,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Response does not match schema",
                    extra={"validation_errors": exc.errors()},
                )
            )

    def _build_error(self, resp: httpx.Response) -> ErrorResponse:
        data = self._parse_json(resp, allow_fail=True)
        return ErrorResponse(
            status=resp.status_code,
            error=data.get("error", ErrorCode.VALIDATION_ERROR),
            message=data.get("message", resp.text),
            extra=data.get("extra", {}),
            trace_id=data.get("trace_id", None),
        )

    # ------------------ Auth ------------------

    def _token_expired(self, token: str | None = None) -> bool:
        token = token if token is not None else self.access_token
        if not token:
            return True
        exp = SletClient._jwt_exp(token)
        # Если exp не прочитать, считаем токен валидным:
        # дальше сработает реактивный refresh.
        return exp is not None and exp - self.token_leeway <= time.time()

    async def get_access_token(self) -> str:
        """Возвращает актуальный access token, обновляя его только при необходимости."""
        if self._token_expired() and self.refresh_token:
            await self.refresh_if_stale(self.access_token)
        return self._auth_headers()["Authorization"].removeprefix("Bearer ")

    async def refresh_if_stale(self, stale_token: str | None) -> None:
        """
        Обновляет токены, только если никто другой не успел это сделать.
        Lock защищает от гонки: при ротации refresh-токена два параллельных
        refresh'а инвалидируют друг друга.
        """
        async with self._refresh_lock:
            if self.access_token != stale_token:
                return  # уже обновлён параллельной задачей
            await self.refresh_tokens()

    async def signin(self, email: str, password: str) -> UserLoginResponse:
        data = await self.request(
            "POST",
            self.api_versions.identify + "/signin",
            schema=UserLoginResponse,
            _skip_refresh=True,
            json={"email": email, "password": password},
        )
        self.access_token = data.access_token
        self.refresh_token = data.refresh_token
        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=200,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing tokens in response",
                )
            )
        return data

    async def signup(
        self,
        name: str,
        email: str,
        password: str,
    ) -> UserRegisterResponse:
        data = await self.request(
            "POST",
            self.api_versions.identify + "/signup",
            schema=UserRegisterResponse,
            _skip_refresh=True,
            json={"name": name, "email": email, "password": password},
        )
        self.access_token = data.access_token
        self.refresh_token = data.refresh_token
        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=200,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing tokens in response",
                )
            )
        return data

    async def refresh_tokens(
        self, refresh_token: str | None = None
    ) -> UserRefreshResponse:
        token = self.refresh_token if refresh_token is None else refresh_token
        data = await self._request_impl(
            "POST",
            self.api_versions.identify + "/refresh",
            schema=UserRefreshResponse,
            _skip_refresh=True,
            json={"refresh_token": token},
        )
        self.access_token = data.access_token
        self.refresh_token = data.refresh_token
        return data

    async def close(self):
        await self._client.aclose()
