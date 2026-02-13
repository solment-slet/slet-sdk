import httpx
import json
from typing import Any
from contextlib import asynccontextmanager

from pydantic import BaseModel, ValidationError

from slet_sdk.schemas import ErrorCode
from slet_sdk.schemas import ErrorResponse
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas.errors import NetworkError
from slet_sdk.schemas.auth import (
    UserLoginResponse,
    UserRegisterResponse,
    UserRefreshResponse,
)

BASE_URL = "http://x.net"


class SafeAsyncClient:
    """
    Обёртка над httpx.AsyncClient, которая автоматически бросает кастомные исключения
    при ошибках сети, статусах и некорректном JSON.
    """

    def __init__(self, client: httpx.AsyncClient):
        self._client = client

    # Проксируем все атрибуты, чтобы можно было работать как с обычным AsyncClient
    def __getattr__(self, item):
        return getattr(self._client, item)

    async def _handle_response(self, resp: httpx.Response):
        """
        Обрабатывает HTTP-ответ от сервера, парсит JSON и превращает ошибки в SletClientError.

        Этот метод выполняет следующие шаги:
        1. Пытается разобрать тело ответа как JSON.
        - Если JSON некорректен, выбрасывается SletClientError с кодом VALIDATION_ERROR
            и полем `extra`, содержащим текст исключения.
        2. Если HTTP-статус успешный (2xx), возвращает распарсенный JSON как словарь.
        3. Если HTTP-статус неуспешный, пытается превратить JSON в модель ErrorResponse.
        - Если JSON не соответствует модели ErrorResponse, создаётся SletClientError
            с VALIDATION_ERROR и подробностями ошибки в поле `extra`.

        Args:
            resp (httpx.Response): Асинхронный HTTP-ответ от сервера.

        Returns:
            dict: Распарсенный JSON из ответа, если статус 2xx.

        Raises:
            SletClientError: Всегда выбрасывается, если ответ содержит ошибку сервера,
                            JSON некорректен или не соответствует схеме ErrorResponse.
                            Поле `extra` всегда является словарём с деталями ошибки.
        """

        try:
            data = resp.json()
        except json.decoder.JSONDecodeError as exc:
            raise SletClientError(
                ErrorResponse(
                    status=resp.status_code,
                    error=ErrorCode.VALIDATION_ERROR,
                    message=f"The server response is not valid JSON: {resp.text}",
                    extra={"exception": str(exc)},  # всегда словарь
                )
            )

        if resp.is_success:
            return data

        # если статус не успешный, пытаемся превратить в ErrorResponse
        try:
            err = ErrorResponse.model_validate(data)
        except ValidationError as exc:
            err = ErrorResponse(
                status=resp.status_code,
                error=ErrorCode.VALIDATION_ERROR,
                message="The server error does not match ErrorResponse pattern!",
                extra=(
                    exc.errors() if hasattr(exc, "model_dump") else {"error": str(exc)}
                ),
                # model_dump() в Pydantic 2 даёт словарь с деталями ошибки
            )

        raise SletClientError(err)

    async def _request(
        self,
        method: str,
        url: str,
        schema: BaseModel | None = None,
        handle: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Универсальный асинхронный HTTP-запрос.

        Args:
            method (str): HTTP-метод ("GET", "POST", "PUT", "PATCH", "DELETE").
            url (str): URL или путь запроса.
            schema (BaseModel, optional): Схема для валидации тела запроса в случае успеха.
            handle (bool, optional): Если True, обрабатывает ответ через _handle_response. Defaults to True.
            **kwargs: Дополнительные параметры, которые передаются в httpx.AsyncClient.

        Returns:
            httpx.Response: Ответ от сервера (если handle=False) или исключение SletClientError при ошибке.
        """
        try:
            response = await self._client.request(method, url, **kwargs)
        except httpx.TransportError as e:
            raise SletClientError(NetworkError(message=str(e)))

        if handle:
            await self._handle_response(response)

        if schema:
            # Пробуем превратить JSON в модель переданную в schema
            try:
                schema.model_validate(response.json())
            except ValidationError as exc:
                raise SletClientError(
                    ErrorResponse(
                        status=response.status_code,
                        error=ErrorCode.VALIDATION_ERROR,
                        message="The server response does not match schema!",
                        extra=exc.errors(),
                    )
                )

        return response

    # Обертки для асинхронных запросов
    async def get(self, url: str, handle: bool = True, **kwargs: Any) -> httpx.Response:
        """Асинхронный GET-запрос с обработкой ошибок через _handle_response."""
        return await self._request("GET", url, handle=handle, **kwargs)

    async def post(
        self,
        url: str,
        schema: BaseModel | None = None,
        handle: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """Асинхронный POST-запрос с обработкой ошибок через _handle_response."""
        return await self._request("POST", url, schema=schema, handle=handle, **kwargs)

    async def put(
        self,
        url: str,
        schema: BaseModel | None = None,
        handle: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """Асинхронный PUT-запрос с обработкой ошибок через _handle_response."""
        return await self._request("PUT", url, schema=schema, handle=handle, **kwargs)

    async def patch(
        self,
        url: str,
        schema: BaseModel | None = None,
        handle: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """Асинхронный PATCH-запрос с обработкой ошибок через _handle_response."""
        return await self._request("PATCH", url, schema=schema, handle=handle, **kwargs)

    async def delete(
        self,
        url: str,
        schema: BaseModel | None = None,
        handle: bool = True,
        **kwargs: Any,
    ) -> httpx.Response:
        """Асинхронный DELETE-запрос с обработкой ошибок через _handle_response."""
        return await self._request(
            "DELETE", url, schema=schema, handle=handle, **kwargs
        )

    @asynccontextmanager
    async def stream(self, method: str, url: str, **kwargs):
        """
        Асинхронный контекстный менеджер для потокового HTTP-запроса.

        Используется для больших ответов без полного чтения в память.
        Проверяет статус ответа и кидает SletClientError при ошибках.
        """
        try:
            async with self._client.stream(method, url, **kwargs) as response:
                # Если статус ошибки
                if not (200 <= response.status_code < 300):
                    # Читаем тело в память только при ошибке
                    try:
                        data = await response.aread()  # читаем весь поток
                        # пытаемся превратить в ErrorResponse
                        try:
                            err_json = json.loads(data)
                            err = ErrorResponse.model_validate(err_json)
                        except (json.JSONDecodeError, ValidationError) as exc:
                            err = ErrorResponse(
                                status=response.status_code,
                                error=ErrorCode.VALIDATION_ERROR,
                                message=f"Streamed response error: {data.decode(errors='ignore')}",
                                extra={"exception": str(exc)},
                            )
                        raise SletClientError(err)
                    except Exception as e:
                        # На случай совсем непредвиденных ошибок
                        raise SletClientError(
                            ErrorResponse(
                                status=response.status_code,
                                error=ErrorCode.VALIDATION_ERROR,
                                message=str(e),
                            )
                        )

                yield response

        except httpx.TransportError as e:
            raise SletClientError(NetworkError(message=str(e)))


class SletClient:
    def __init__(
        self,
        base_url: str = BASE_URL,  # URL Slet сервера
        timeout: float = 500.0,  # Максимальное время ожидания ответа от сервера
        ssl_verify: bool = True,  # Проверять ли сертификат SSL (разрешает исключительно HTTPS)
    ):
        self.base_url = base_url
        self.raw_client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(timeout), verify=ssl_verify
        )
        self.client = SafeAsyncClient(self.raw_client)  # Оборачиваем в SafeAsyncClient
        self.access_token = None
        self.refresh_token = None

    async def signin(self, email: str, password: str) -> dict:
        response = await self.client.post(
            "/signin",
            schema=UserLoginResponse,
            json={"email": email, "password": password},
        )
        data = response.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")

        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=response.status_code,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing access or refresh token in response",
                )
            )

        return data

    async def signup(self, name: str, email: str, password: str) -> dict:
        """
        Регистрация пользователя в Slet.
        Сохраняет токены для доступа к API.
        """
        response = await self.client.post(
            "/signup",
            schema=UserRegisterResponse,
            json={"name": name, "email": email, "password": password},
        )

        data = response.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")

        if not self.access_token or not self.refresh_token:
            raise SletClientError(
                ErrorResponse(
                    status=response.status_code,
                    error=ErrorCode.VALIDATION_ERROR,
                    message="Missing access or refresh token in response",
                )
            )

        return data

    async def refresh_tokens(self):
        response = await self.client.post(
            "/refresh",
            schema=UserRefreshResponse,
            json={"refresh_token": self.refresh_token},
        )
        data = response.json()
        self.access_token = data.get("access_token")
        self.refresh_token = data.get("refresh_token")
        return data

    def _auth_headers(self):
        if not self.access_token:
            raise ValueError("Access token is missing. Login first.")
        return {"Authorization": f"Bearer {self.access_token}"}

    async def init_llm(self, api_key: str):
        response = await self.client.post(
            "/init", json={"message": api_key}, headers=self._auth_headers()
        )
        return response.json()

    async def update_model_settings(
        self,
        ai_model,
        system_prompt,
        temperature,
        max_tokens_allowed,
        message_max_tokens,
    ):
        payload = "A65BB6###".join(
            [
                ai_model,
                system_prompt.replace("\n", "\\n"),
                str(temperature),
                str(max_tokens_allowed),
                str(message_max_tokens),
            ]
        )
        response = await self.client.post(
            "/update", json={"message": payload}, headers=self._auth_headers()
        )
        return response.json()

    async def create_chat(self):
        response = await self.client.post("/chat/create", headers=self._auth_headers())
        return response.json()

    async def answer(
        self, message: str, chat_id: str | None = None, use_processing: bool = True
    ):
        route = "/answer/serverprocessing" if use_processing else "/answer/answer"
        payload = {"message": message}
        if chat_id:
            payload["chatId"] = chat_id
        response = await self.client.post(
            route, json=payload, headers=self._auth_headers()
        )
        return response.json()

    async def generate_with_tools(
        self, message: str, chat_id: str | None = None, stream: bool = False
    ):
        route = "/generate/tools"

        payload = {"message": message, "stream": stream}

        if chat_id:
            payload["chatId"] = chat_id

        # ================= NON-STREAM MODE =================
        if not stream:
            response = await self.client.post(
                route, json=payload, headers=self._auth_headers()
            )
            response.raise_for_status()
            return response.json()

        # ================= STREAM MODE (SSE + JSON) =================
        async def sse_generator():
            async with self.client.stream(
                "POST",
                route,
                json=payload,
                headers={
                    **self._auth_headers(),
                    "Accept": "text/event-stream",
                },
            ) as response:

                response.raise_for_status()

                buffer = ""

                async for raw in response.aiter_raw():
                    buffer += raw.decode("utf-8")

                    # SSE event заканчивается двойным переводом строки
                    while "\r\n\r\n" in buffer or "\n\n" in buffer:
                        if "\r\n\r\n" in buffer:
                            event, buffer = buffer.split("\r\n\r\n", 1)
                        else:
                            event, buffer = buffer.split("\n\n", 1)

                        for line in event.splitlines():
                            if not line.startswith("data:"):
                                continue

                            data = line[5:]
                            if data.startswith(" "):
                                data = data[1:]  # убрать служебный пробел SSE

                            # Конец стрима
                            if data == "[DONE]":
                                return

                            # JSON payload
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            token = obj.get("token")
                            if token is not None:
                                yield token

        return sse_generator()

    async def generate_with_create(
        self,
        message: str,
        system_message: str,
        ai_model: str,
        ai_temperature: float,
        max_tokens: int,
        stream: bool = False,
    ):
        route = "/generate/create"

        payload = {
            "message": message,
            "systemMessage": system_message,
            "aiModel": ai_model,
            "aiTemperature": str(ai_temperature),
            "maxTokens": str(max_tokens),
            "stream": stream,
        }

        # ================= NON-STREAM MODE =================
        if not stream:
            response = await self.client.post(
                route, json=payload, headers=self._auth_headers()
            )
            response.raise_for_status()
            return response.json()

        # ================= STREAM MODE (SSE + JSON) =================
        async def sse_generator():
            async with self.client.stream(
                "POST",
                route,
                json=payload,
                headers={
                    **self._auth_headers(),
                    "Accept": "text/event-stream",
                },
            ) as response:

                response.raise_for_status()

                buffer = ""

                async for raw in response.aiter_raw():
                    buffer += raw.decode("utf-8")

                    # SSE event заканчивается двойным переводом строки
                    while "\r\n\r\n" in buffer or "\n\n" in buffer:
                        if "\r\n\r\n" in buffer:
                            event, buffer = buffer.split("\r\n\r\n", 1)
                        else:
                            event, buffer = buffer.split("\n\n", 1)

                        for line in event.splitlines():
                            if not line.startswith("data:"):
                                continue

                            data = line[5:]
                            if data.startswith(" "):
                                data = data[1:]  # убрать служебный пробел SSE

                            # Конец стрима
                            if data == "[DONE]":
                                return

                            # JSON payload
                            try:
                                obj = json.loads(data)
                            except json.JSONDecodeError:
                                continue

                            token = obj.get("token")
                            if token is not None:
                                yield token

        return sse_generator()

    async def get_chat_history(self, chat_id: str):
        response = await self.client.get(
            f"/chat/history/{chat_id}", headers=self._auth_headers()
        )
        return response.json()

    async def delete_chat(self, chat_id: str):
        response = await self.client.delete(
            f"/chat/delete/{chat_id}", headers=self._auth_headers()
        )
        return response.status_code == 204

    async def close(self):
        await self.client.aclose()
