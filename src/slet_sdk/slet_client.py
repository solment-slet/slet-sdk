import httpx
import json
import logging
from typing import (
    Any,
    Optional,
    Type,
    TypeVar,
    overload,
)

from pydantic import BaseModel, ValidationError

from slet_sdk.core.typing import LoggerLike
from slet_sdk.core.schemas import ErrorCode, ErrorResponse
from slet_sdk.core.exceptions import SletClientError
from slet_sdk.core.schemas.errors import NetworkError
from slet_sdk.core.schemas.auth import (
    UserLoginResponse,
    UserRegisterResponse,
    UserRefreshResponse,
)
from slet_sdk.aelite.resources.resource import AeliteResource

T = TypeVar("T", bound=BaseModel)

BASE_URL = "http://x.net"


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
        base_url: str = BASE_URL,
        timeout: float = 500.0,
        ssl_verify: bool = True,
        logger: Optional[LoggerLike] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.base_url = base_url
        self.base_ws_url = self.base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
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
            raise SletClientError(
                ErrorResponse(
                    status=0,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Access token is missing. Call signin() first.",
                )
            )
        return {"Authorization": f"Bearer {self.access_token}"}

    @overload
    async def request(
            self, method: str, url: str, schema: Type[T], **kwargs: Any
    ) -> T:
        ...

    @overload
    async def request(
            self, method: str, url: str, schema: None = None, **kwargs: Any
    ) -> dict:
        ...

    async def request(
            self,
            method: str,
            url: str,
            schema: Type[T] | None = None,
            **kwargs: Any,
    ) -> dict | T:
        return await self._request_impl(method, url, schema=schema, **kwargs)

    async def _request_impl(
            self,
            method: str,
            url: str,
            schema: Type[T] | None = None,
            *,
            _skip_refresh: bool = False,
            **kwargs: Any,
    ) -> dict | T:
        last_resp: httpx.Response | None = None

        for attempt in range(2):
            merged_headers = {**kwargs.get("headers", {})}
            if self.access_token:
                merged_headers["Authorization"] = f"Bearer {self.access_token}"
            request_kwargs = {**kwargs, "headers": merged_headers}

            try:
                resp = await self._client.request(method, url, **request_kwargs)
            except httpx.TransportError as e:
                raise SletClientError(NetworkError(message=str(e)))

            last_resp = resp

            if 200 <= resp.status_code < 300:
                data = self._parse_json(resp)
                return self._validate_schema(data, schema, resp.status_code) if schema else data

            if (
                    resp.status_code == 401
                    and not _skip_refresh
                    and attempt == 0
                    and self.refresh_token
            ):
                err_data = self._parse_json(resp, allow_fail=True)
                if err_data.get("error") == ErrorCode.INVALID_ACCESS_TOKEN:
                    await self.refresh_tokens()
                    continue

            raise SletClientError(self._build_error(resp))

        assert last_resp is not None
        raise SletClientError(self._build_error(last_resp))

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

    def _validate_schema(self, data: dict, schema: Type[T], status: int) -> T:
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

    async def signin(self, email: str, password: str) -> UserLoginResponse:
        data = await self.request(
            "POST",
            "/signin",
            schema=UserLoginResponse,
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

    async def signup(self, name: str, email: str, password: str) -> UserRegisterResponse:
        data = await self.request(
            "POST",
            "/signup",
            schema=UserRegisterResponse,
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

    async def refresh_tokens(self, refresh_token: str | None = None) -> UserRefreshResponse:
        token = self.refresh_token if refresh_token is None else refresh_token
        data = await self._request_impl(
            "POST",
            "/refresh",
            schema=UserRefreshResponse,
            _skip_refresh=True,
            json={"refresh_token": token},
        )
        self.access_token = data.access_token
        self.refresh_token = data.refresh_token
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
