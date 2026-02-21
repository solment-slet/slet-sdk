import httpx
import json
from typing import Any, Dict
from pydantic import BaseModel, ValidationError

from slet_sdk.schemas import ErrorCode, ErrorResponse
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas.errors import NetworkError
from slet_sdk.agent import AgentSession
from slet_sdk.schemas.agent import AgentManifest
from slet_sdk.schemas.auth import (
    UserLoginResponse,
    UserRegisterResponse,
    UserRefreshResponse,
)

BASE_URL = "http://x.net"


class SletClient:
    """
    Лёгкий асинхронный HTTP-клиент для Slet API с авто-рефрешем токена.
    """

    def __init__(
        self, base_url: str = BASE_URL, timeout: float = 500.0, ssl_verify: bool = True
    ):
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout), verify=ssl_verify
        )
        self.access_token: str | None = None
        self.refresh_token: str | None = None

    # ------------------ Context Manager ------------------
    async def __aenter__(self) -> "SletClient":
        # Просто возвращаем себя
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Закрываем внутренний httpx клиент
        await self.close()

    # ------------------ Internal Methods ------------------
    def _auth_headers(self) -> dict[str, str]:
        if not self.access_token:
            raise ValueError("Access token is missing. Login first.")
        return {"Authorization": f"Bearer {self.access_token}"}

    async def _request(
        self, method: str, url: str, schema: BaseModel | None = None, **kwargs: Any
    ) -> dict:
        """
        Универсальный запрос с автоматическим рефрешем токена.
        """
        for attempt in range(2):
            # добавляем авторизацию, если есть токен
            headers = kwargs.pop("headers", {})
            if self.access_token:
                headers.update(self._auth_headers())
            kwargs["headers"] = headers

            try:
                resp = await self._client.request(method, url, **kwargs)
            except httpx.TransportError as e:
                raise SletClientError(NetworkError(message=str(e)))

            # успешный ответ
            if 200 <= resp.status_code < 300:
                data = self._parse_json(resp)
                if schema:
                    self._validate_schema(data, schema, resp.status_code)
                return data

            # 401 → проверяем рефреш токена
            if resp.status_code == 401:
                data = self._parse_json(resp, allow_fail=True)
                if data.get("error") == ErrorCode.INVALID_ACCESS_TOKEN:
                    if self.refresh_token:
                        await self.refresh_tokens()
                        continue  # повторяем запрос после рефреша
                # иначе падаем с ошибкой
            # Любой другой ответ → выбрасываем
            raise SletClientError(self._build_error(resp))

        # если дошли сюда, значит 2 попытки не удались
        raise SletClientError(self._build_error(resp))

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

    def _validate_schema(self, data: dict, schema: BaseModel, status: int):
        try:
            schema.model_validate(data)
        except ValidationError as exc:
            raise SletClientError(
                ErrorResponse(
                    status=status,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Response does not match schema",
                    extra=exc.errors(),
                )
            )

    def _build_error(self, resp: httpx.Response) -> ErrorResponse:
        data = self._parse_json(resp, allow_fail=True)
        return ErrorResponse(
            status=resp.status_code,
            error=data.get("error", ErrorCode.VALIDATION_ERROR),
            message=data.get("message", resp.text),
            extra=data.get("extra", {}),
        )

    # ------------------ Public API ------------------

    async def signin(self, email: str, password: str) -> dict:
        data = await self._request(
            "POST",
            "/signin",
            schema=UserLoginResponse,
            json={"email": email, "password": password},
        )
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=200,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing tokens in response",
                )
            )
        return data

    async def signup(self, name: str, email: str, password: str) -> dict:
        data = await self._request(
            "POST",
            "/signup",
            schema=UserRegisterResponse,
            json={"name": name, "email": email, "password": password},
        )
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=200,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing tokens in response",
                )
            )
        return data

    async def refresh_tokens(self, refresh_token: str | None = None) -> dict:
        token = self.refresh_token if refresh_token is None else refresh_token
        data = await self._request(
            "POST",
            "/refresh",
            schema=UserRefreshResponse,
            json={"refresh_token": token},
        )
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    async def deploy_agent(
        self, thread_id: str, manifest: AgentManifest | Dict[str, Any]
    ) -> dict:
        """
        Деплоит агента на сервере.
        :param thread_id: ID сессии (чата)
        :param manifest: Конфигурацией агента (AgentManifest)
        """
        if not isinstance(manifest, dict):
            manifest = manifest.model_dump()

        print(f"[DEBUG] {str(manifest)}")

        return await self._request(
            "POST",
            f"/ae/agent/deploy/{thread_id}",
            json=manifest,
        )

    async def connect_agent(self, thread_id: str) -> AgentSession:
        """
        Создает WebSocket сессию для общения с агентом.
        :param thread_id: ID сессии
        :return: Объект AgentSession
        """
        # Формируем WS URL на основе HTTP URL
        # http://x.net -> ws://x.net
        # https://x.net -> wss://x.net
        ws_base = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_base}/ae/agent/ws/{thread_id}"

        # Получаем заголовки авторизации
        headers = {}
        if self.access_token:
            # Websockets библиотека требует список кортежей или dict, но заголовки Auth часто передают в query params
            # или через extra_headers. AElite сервер должен поддерживать Auth header.
            headers["Authorization"] = f"Bearer {self.access_token}"

        session = AgentSession(ws_url, headers)
        await session.connect()
        return session

    async def get(self, url: str, **kwargs: Any) -> dict:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> dict:
        return await self._request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> dict:
        return await self._request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> dict:
        return await self._request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> dict:
        return await self._request("DELETE", url, **kwargs)

    async def close(self):
        await self._client.aclose()
