from __future__ import annotations
from dataclasses import dataclass
import json
import asyncio
import logging
import mimetypes
from pathlib import Path
import inspect
from typing import (
    AsyncGenerator,
    AsyncIterable,
    Callable,
    Optional,
    Dict,
    Any,
    List,
    Union,
    Awaitable,
    TYPE_CHECKING,
)

import websockets
from websockets.exceptions import ConnectionClosed
from slet_sdk.core.schemas import ErrorResponse, ErrorCode
from slet_sdk.core.schemas.errors import CallbackError, NetworkError
from slet_sdk.core.exceptions import SletClientError

if TYPE_CHECKING:
    from slet_sdk.aelite.resources.resource import AeliteResource
    from slet_sdk.aelite.manifest import AgentManifest

@dataclass
class _End:
    pass

@dataclass
class _Error:
    error: ErrorResponse

# Тип элемента очереди
_QueueItem = Union[str, _End, _Error]


class AgentSession:
    """
    Сессия общения с агентом через WebSocket.

    Особенности:
    - Фоновая задача (listener) запускается автоматически при вызове connect().
    - Поддерживает режим "Запрос-Ответ" (chat, stream).
    - Поддерживает инициативу агента (незапрошенные сообщения).
    - Автоматически обрабатывает вызовы инструментов (client tools).
    """

    def __init__(
            self,
            base_url: str,
            thread_id: str,
            headers: dict,
            manifest: AgentManifest | None = None,
            resource: AeliteResource | None = None,
    ):
        self.thread_id = thread_id
        self.manifest = manifest
        self._resource = resource
        self.sub: dict[str, "AgentSession"] = {}

        ws_base = base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        base_url = base_url.replace("ws://", "http://").replace(
            "wss://", "https://"
        )
        self.http_url = base_url
        self.ws_url = ws_base
        self._ws_connect_url = f"{self.ws_url}/ae/agent/ws/" + self.thread_id
        self.headers = headers
        self.websocket = None
        self.logger = getattr(self._resource, "logger", logging.getLogger(__name__))
        self._is_connected = False

        # --- Настройки ---
        # Если True: незапрошенные сообщения приходят в on_incoming_stream (как асинхронный итератор).
        # Если False: незапрошенные сообщения буферизуются и приходят целиком в on_message.
        self.stream_unsolicited = False

        # --- Колбэки ---
        # 1. on_message(text: str): Вызывается для полных незапрошенных сообщений.
        # Может быть как sync, так и async функцией.
        self.on_message: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None

        # 2. on_incoming_stream(iterator): Вызывается при начале незапрошенного сообщения (если stream_unsolicited=True).
        # Используем AsyncIterable, чтобы PyCharm не ругался на типы генераторов.
        self.on_incoming_stream: Optional[Callable[[AsyncIterable[str]], Union[None, Awaitable[None]]]] = None

        # 3. on_tool_start(name: str): Уведомление о старте инструмента на сервере.
        self.on_tool_start: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None

        # 4. on_error(error: str): Глобальная обработка ошибок фонового слушателя.
        self.on_error: Optional[Callable[[ErrorResponse], Union[None, Awaitable[None]]]] = None

        # 5. on_model_change(provider: str, model: str): Уведомление о смене глобальной LLM модели/провайдера на сервере.
        self.on_model_change: Optional[Callable[[str, str], Union[None, Awaitable[None]]]] = None

        # Реестр инструментов
        self._registered_tools: Dict[str, Callable] = {}

        # --- Внутренние механизмы ---
        self._listener_task: Optional[asyncio.Task] = None

        # Очередь для ответов на активный запрос пользователя (chat/stream)
        self._response_queue: asyncio.Queue[_QueueItem] = asyncio.Queue()

        # Очередь для стриминга незапрошенных сообщений (если stream_unsolicited=True)
        self._unsolicited_stream_queue: Optional[asyncio.Queue[_QueueItem]] = None

        # Буфер для накопления текста незапрошенного сообщения (если stream_unsolicited=False)
        self._unsolicited_buffer: List[str] = []

        # Флаг: мы ждем ответ на явный запрос пользователя?
        self._waiting_for_response = False

        # Флаг: прямо сейчас идет прием данных от агента (любых)
        # Нужен для блокировки отправки новых сообщений, пока агент не закончит текущую мысль.
        self._is_incoming_traffic = False

        # --- Настройки реконнекта ---
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5  # секунд
        self._reconnect_attempt = 0
        self._manual_disconnect = False  # чтобы отличать ручное закрытие
        self._reconnect_event = asyncio.Event()
        self._reconnect_event.set()

        # Блокировка, чтобы нельзя было вызвать chat/stream параллельно
        self._lock = asyncio.Lock()

    async def connect(self):
        """Устанавливает соединение и запускает фоновый слушатель."""
        try:
            self.websocket = await websockets.connect(
                self._ws_connect_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                additional_headers=self.headers,
            )
            self._is_connected = True

            # АВТОМАТИЧЕСКИЙ ЗАПУСК ФОНОВОЙ ЗАДАЧИ
            self._listener_task = asyncio.create_task(self._listen_loop())

            self.logger.info(f"Connected to agent at {self.ws_url}")
        except websockets.exceptions.InvalidStatus as e:
            self.logger.error(f"Failed to connect to agent: {e}")

            err = ErrorResponse(
                status=e.response.status_code,
                error=ErrorCode.WEBSOCKET_ERROR,
                message=str(e),
            )
            self._invoke_callback(self.on_error, err)
            raise SletClientError(err)

    async def disconnect(self):
        """Останавливает фоновые задачи и закрывает сокет."""
        self._manual_disconnect = True
        self._is_connected = False

        # Отменяем фонового слушателя
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from agent")

    async def disconnect_all(self):
        """Рекурсивно отключает всё дерево подагентов."""
        for sub_session in self.sub.values():
            await sub_session.disconnect_all()
        await self.disconnect()

    async def redeploy(self, new_manifest: "AgentManifest | None" = None):
        """
        Пересоздаёт этого агента на сервере, сохраняя thread_id (историю).
        Если new_manifest не передан — редеплоит текущий манифест.
        """
        if self._resource is None:
            raise RuntimeError(
                "Cannot redeploy: session was created without resource reference. "
                "Use client.aelite.deploy_and_connect() to get a redeployable session."
            )

        manifest = new_manifest or self.manifest
        if manifest is None:
            raise ValueError("No manifest provided and no stored manifest")

        # Обновляем сохранённый манифест
        self.manifest = manifest

        # Вызываем deploy на сервере с существующим thread_id
        await self._resource.deploy(manifest, thread_id=self.thread_id)

    def register_tools(self, tools: List[Callable]):
        """Регистрирует функции инструментов."""
        for fn in tools:
            name = getattr(fn, '_tool_name', None) or fn.__name__
            self._registered_tools[name] = fn
            self.logger.debug(f"Registered client tool: {name}")

    def register_tools_from_registry(self):
        """Загружает инструменты из глобального реестра SDK (если есть)."""
        try:
            from slet_sdk.aelite.tools import get_registered_tools
            self._registered_tools.update(get_registered_tools())
        except ImportError:
            pass

    async def send(self, message: str):
        """Низкоуровневая отправка сообщения в сокет."""

        # Если идет реконнект — ждем
        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(
                NetworkError(message="WebSocket is not connected")
            )

        await self.websocket.send(message)

    # ------------------------------------------------------------------
    # Публичные методы запросов
    # ------------------------------------------------------------------

    async def stream(
            self,
            message: str,
            files: Optional[list[Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Отправляет сообщение агенту и возвращает генератор с ответом.
        Блокируется, если агент в данный момент уже передает какое-то сообщение.
        """
        # Отправляем файлы на сервер и формируем payload
        if files:
            # 1. Загружаем файлы на сервер Slet
            attachments: list[dict] = []
            for f in files:
                res = await self.upload_file(f)
                attachments.append({"ref": f"upload:{res['file_id']}"})

            # 2. Формируем payload
            message = json.dumps({
                "type": "message",
                "text": message,
                "attachments": attachments
            }) if attachments else message

        # Ждем завершения реконнекта
        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(
                NetworkError(message="Cannot send message: not connected")
            )

        # Блокируем сессию, чтобы другие вызовы chat/stream ждали очереди
        async with self._lock:
            # ЗАЩИТА ОТ ГОНКИ:
            # Если прямо сейчас летит фоновое сообщение (инициатива агента),
            # ждем его завершения (пока не придет end_of_answer).
            # Иначе наш запрос смешается с хвостом фонового ответа.
            while self._is_incoming_traffic:
                await asyncio.sleep(0.05)

            # Очищаем очередь от старого мусора (на всякий случай)
            while not self._response_queue.empty():
                self._response_queue.get_nowait()

            # Включаем режим ожидания ответа
            self._waiting_for_response = True

            try:
                await self.send(message)

                # Читаем из очереди, которую наполняет _listen_loop
                while True:
                    item = await self._response_queue.get()

                    match item:
                        case str():
                            yield item
                        case _End():
                            break
                        case _Error(error):
                            raise SletClientError(error)

            finally:
                # Выключаем режим ожидания
                self._waiting_for_response = False

    async def chat(
        self,
        message: str,
        files: Optional[list[Any]] = None,
    ) -> str:
        """
        Отправляет сообщение и возвращает полный текстовый ответ.
        Работает через stream(), собирая чанки в строку.
        """
        full_response = []
        async for chunk in self.stream(
                message=message,
                files=files
        ):
            full_response.append(chunk)
        return "".join(full_response)

    async def upload_file(self, file: Any) -> dict:
        """
        Загружает файл на сервер через HTTP.

        Принимает:
          - str          — текстовое содержимое (отправляется как text/plain)
          - bytes        — сырые байты
          - file-like    — объект с .read() (BufferedReader, SpooledTemporaryFile, и т.д.)
        """
        # ═══ Нормализуем входные данные ═══

        if isinstance(file, str):
            file_data = file.encode("utf-8")
            filename = "text.txt"
            mime_type = "text/plain"

        elif isinstance(file, bytes):
            file_data = file
            filename = "file.bin"
            mime_type = "application/octet-stream"

        elif hasattr(file, "read"):
            raw_name = getattr(file, "name", None) or getattr(file, "filename", None)
            if raw_name:
                filename = Path(raw_name).name
            else:
                filename = "upload.bin"

            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

            if asyncio.iscoroutinefunction(getattr(file, "read", None)):
                file_data = await file.read()
            else:
                file_data = file.read()

        else:
            raise TypeError(
                f"Unsupported file type: {type(file).__name__}. "
                f"Expected str, bytes, or file-like object."
            )

        # ═══ Отправляем через self.request (httpx) ═══

        upload_url = f"{self.http_url}/ae/upload/"

        return await self._resource._request(
            "POST",
            upload_url,
            files={"file": (filename, file_data, mime_type)},
            data={"mime_type": mime_type},
        )

    async def trigger(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Отправляет триггер (событие) агенту не ожидая ответ.
        Для получения ответов можно использовать коллбэки (on_message, on_incoming_stream)
        """
        if not self._is_connected:
            raise ConnectionError("Not connected")

        event = {"type": "trigger", "name": name, "payload": payload or {}}

        await self.send(json.dumps(event))

    # ------------------------------------------------------------------
    # Приватный фоновый цикл (Listen Loop)
    # ------------------------------------------------------------------

    async def _listen_loop(self):
        """
        Постоянно читает вебсокет. Маршрутизирует сообщения:
        1. Системные события -> Обработчики.
        2. Токены (ответ на chat) -> _response_queue.
        3. Токены (инициатива агента) -> Буфер или _unsolicited_stream_queue.
        """
        self.logger.debug("Background listener started")
        try:
            async for raw_msg in self.websocket:
                is_system_msg = False

                # Оптимизация: проверяем '{', чтобы не парсить каждый токен текста как JSON
                if raw_msg.startswith("{"):
                    try:
                        data = json.loads(raw_msg)
                        if isinstance(data, dict) and "type" in data:
                            event_type = data["type"]

                            # --- КОНЕЦ ОТВЕТА ---
                            if event_type == "end_of_answer":
                                is_system_msg = True
                                # Линия свободна
                                self._is_incoming_traffic = False

                                # Сценарий А: Мы ждали ответ на chat/stream
                                if self._waiting_for_response:
                                    await self._response_queue.put(_End())

                                # Сценарий Б: Это конец незапрошенного сообщения
                                else:
                                    if self.stream_unsolicited:
                                        # Завершаем стрим
                                        if self._unsolicited_stream_queue:
                                            await self._unsolicited_stream_queue.put(_End())
                                            self._unsolicited_stream_queue = None
                                    else:
                                        # Отдаем буфер
                                        if self._unsolicited_buffer:
                                            full_text = "".join(self._unsolicited_buffer)
                                            self._unsolicited_buffer.clear()
                                            self._invoke_callback(self.on_message, full_text)
                                continue

                            # --- ВЫЗОВ ИНСТРУМЕНТА ---
                            if event_type == "client_tool_call":
                                is_system_msg = True
                                asyncio.create_task(self._handle_client_tool_call(data["payload"]))
                                continue

                            # --- СТАРТ ИНСТРУМЕНТА (Уведомление) ---
                            if event_type == "tool_start":
                                is_system_msg = True
                                name = data.get("payload", {}).get("name")
                                self._invoke_callback(self.on_tool_start, name)
                                continue

                            # --- СМЕНА МОДЕЛИ/ПРОВАЙДЕРА (Уведомление) ---
                            if event_type == "model_changed":
                                is_system_msg = True
                                provider = data.get("payload", {}).get("provider")
                                model = data.get("payload", {}).get("model")
                                self._invoke_callback(self.on_model_change, provider, model)
                                continue

                            if event_type == "error":
                                is_system_msg = True
                                self._is_incoming_traffic = False

                                # Очистить незавершённый unsolicited стрим/буфер:
                                if self._unsolicited_stream_queue:
                                    await self._unsolicited_stream_queue.put(_End())
                                    self._unsolicited_stream_queue = None
                                self._unsolicited_buffer.clear()

                                err = ErrorResponse(**data.get("payload", {}))
                                if self._waiting_for_response:
                                    # Если кто-то ожидает ответ, кладём ошибку в очередь
                                    await self._response_queue.put(_Error(err))
                                self._invoke_callback(self.on_error, err)
                                continue


                    except json.JSONDecodeError:
                        # Сообщение начиналось с {, но не является валидным JSON (редкий случай в тексте)
                        pass

                # --- ОБРАБОТКА ТЕКСТА (ТОКЕНОВ) ---
                if not is_system_msg:
                    # Сценарий А: Активный диалог
                    if self._waiting_for_response:
                        await self._response_queue.put(raw_msg)

                    # Сценарий Б: Инициатива агента (незапрошенное сообщение)
                    else:
                        # Поднимаем флаг трафика, чтобы chat() не вклинился в середину
                        self._is_incoming_traffic = True

                        if self.stream_unsolicited:
                            # Режим стриминга:
                            # Если это первый токен - создаем очередь и уведомляем пользователя
                            if self._unsolicited_stream_queue is None:
                                self._unsolicited_stream_queue = asyncio.Queue()
                                gen = self._make_unsolicited_generator(self._unsolicited_stream_queue)
                                self._invoke_callback(self.on_incoming_stream, gen)

                            # Кладем токен
                            await self._unsolicited_stream_queue.put(raw_msg)
                        else:
                            # Режим буферизации:
                            self._unsolicited_buffer.append(raw_msg)

        except ConnectionClosed as e:
            self.logger.warning(f"WebSocket closed: {e}")
            self._is_connected = False
            self._cleanup_queues()

            if not self._manual_disconnect:
                error = NetworkError(message=f"WebSocket connection closed: {e}")
                if self.on_error:
                    self._invoke_callback(self.on_error, error)

                asyncio.create_task(self._reconnect_loop())

        except Exception as e:
            self.logger.error(f"Error in background listener: {e}")
            self._is_connected = False
            self._cleanup_queues()

            if not self._manual_disconnect:
                error = NetworkError(message=f"Listener crashed: {e}")
                if self.on_error:
                    self._invoke_callback(self.on_error, error)

                asyncio.create_task(self._reconnect_loop())

    def _cleanup_queues(self):
        """Освобождает всех ожидающих при разрыве соединения и очищаем буфера."""
        if self._waiting_for_response:
            self._response_queue.put_nowait(_End())
        if self._unsolicited_stream_queue:
            self._unsolicited_stream_queue.put_nowait(_End())
            self._unsolicited_stream_queue = None
        self._unsolicited_buffer.clear()
        self._is_incoming_traffic = False

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _invoke_callback(self, callback: Callable, *args):
        """
        Безопасно вызывает колбэк.
        Если callback - корутина, планирует её выполнение в Event Loop.
        Если callback - синхронная функция, вызывает её сразу.
        """
        if not callback:
            return

        try:
            if inspect.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args))
            else:
                # Для синхронных функций.
                # Если функция тяжелая, она может заблокировать _listen_loop!
                self.logger.debug(f"{callback.__name__} is not a coroutine function. Make sure that it does not perform any blocking actions.")
                callback(*args)
        except Exception as e:
            self.logger.error(f"Error invoking callback {callback}: {e}")
            # Вызов on_error, если это не он сам
            if self.on_error is not None and callback is not self.on_error:
                try:
                    if inspect.iscoroutinefunction(self.on_error):
                        asyncio.create_task(self.on_error(CallbackError(message=str(e))))
                    else:
                        self.on_error(CallbackError(message=str(e)))
                except Exception as err:
                    # Предотвращаем рекурсию или крэш
                    self.logger.error(f"Error in on_error callback: {err}")

    async def _make_unsolicited_generator(self, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Генератор, который читает из очереди незапрошенных сообщений."""
        while True:
            item = await queue.get()
            match item:
                case str():
                    yield item
                case _End():
                    break
                case _Error(error):
                    raise SletClientError(error)

    async def _handle_client_tool_call(self, payload: dict):
        """Обработка вызова клиентского инструмента."""
        tool_name = payload.get("name", "")
        args = payload.get("args", {})
        call_id = payload.get("tool_call_id", "")

        self.logger.debug(f"Handling tool call: {tool_name}")
        fn = self._registered_tools.get(tool_name)

        result = None
        if not fn:
            result = f"Error: Tool '{tool_name}' not registered on client"
            self.logger.warning(result)
        else:
            try:
                if inspect.iscoroutinefunction(fn):
                    result = await fn(**args)
                else:
                    # Запускаем синхронные инструменты в экзекьюторе, чтобы не блокировать loop
                    loop = asyncio.get_running_loop()
                    result = await loop.run_in_executor(None, lambda: fn(**args))
            except Exception as e:
                result = f"Error executing tool: {e}"
                self.logger.error(f"Tool execution failed: {e}")

        # Отправка результата
        response = {
            "type": "client_tool_result",
            "payload": {
                "tool_call_id": call_id,
                "result": str(result),
            }
        }

        if self._is_connected:
            await self.websocket.send(json.dumps(response))

    async def _reconnect_loop(self):
        """
        Пытается переподключиться к WebSocket до self._max_reconnect_attempts.
        Вызывает on_error при каждой неудаче.
        """
        self._reconnect_attempt = 0
        self._reconnect_event.clear()

        while self._reconnect_attempt < self._max_reconnect_attempts:
            self._reconnect_attempt += 1

            try:
                self.logger.info(
                    f"Reconnect attempt {self._reconnect_attempt}/"
                    f"{self._max_reconnect_attempts}"
                )

                self.websocket = await websockets.connect(
                    self._ws_connect_url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=10,
                    additional_headers=self.headers,
                )

                self._is_connected = True
                self._manual_disconnect = False

                # Перезапускаем listener
                self._listener_task = asyncio.create_task(self._listen_loop())
                self._reconnect_event.set()

                self.logger.info("WebSocket successfully reconnected")
                return  # ✅ Успешный реконнект

            except Exception as e:
                is_last = self._reconnect_attempt == self._max_reconnect_attempts

                message = (
                    "WebSocket reconnection failed. Maximum attempts reached."
                    if is_last
                    else f"WebSocket reconnection attempt "
                         f"{self._reconnect_attempt} failed"
                )

                error = NetworkError(message=f"{message}: {e}")

                self.logger.error(message)

                if self.on_error:
                    self._invoke_callback(self.on_error, error)

                if is_last:
                    self._reconnect_event.set()
                    raise SletClientError(error)

                await asyncio.sleep(self._reconnect_delay)