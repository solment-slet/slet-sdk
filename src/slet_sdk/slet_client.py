import httpx
import json
from typing import Any
from pydantic import BaseModel, ValidationError

from slet_sdk.core.schemas import ErrorCode, ErrorResponse
from slet_sdk.core.exceptions import SletClientError
from slet_sdk.core.schemas.errors import NetworkError
from slet_sdk.core.schemas.auth import (
    UserLoginResponse,
    UserRegisterResponse,
    UserRefreshResponse,
)

# Ленивый импорт сервисов
from slet_sdk.aelite.resource import AeliteResource

BASE_URL = "http://x.net"


class SletClient:
    """
    Единый асинхронный клиент для всех сервисов Slet API.

    Использование:
        async with SletClient() as client:
            await client.signin("user@example.com", "password")
            await client.agents.deploy(thread_id, manifest)
            balance = await client.billing.get_balance()
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 500.0,
        ssl_verify: bool = True,
    ):
        self.base_url = base_url
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout), verify=ssl_verify
        )
        self.access_token: str | None = None
        self.refresh_token: str | None = None

        # ── Namespace-ресурсы (продукты) ──────────────────────
        self.aelite = AeliteResource(self)

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

    async def request(
        self,
        method: str,
        url: str,
        schema: BaseModel | None = None,
        **kwargs: Any,
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

    # ------------------ Auth ------------------

    async def signin(self, email: str, password: str) -> dict:
        data = await self.request(
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
        data = await self.request(
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
        data = await self.request(
            "POST",
            "/refresh",
            schema=UserRefreshResponse,
            json={"refresh_token": token},
        )
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    # ──────────────── Generic HTTP ───────────────────────────

    async def get(self, url: str, **kwargs: Any) -> dict:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> dict:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> dict:
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> dict:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> dict:
        return await self.request("DELETE", url, **kwargs)

    async def close(self):
        await self._client.aclose()
