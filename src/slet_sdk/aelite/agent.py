from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
)

from websockets.exceptions import ConnectionClosed, InvalidStatus

from slet_sdk.aelite.tools import get_registered_tools
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas import ErrorResponse, ErrorCode
from slet_sdk.schemas.errors import CallbackError, NetworkError
from slet_sdk.aelite.typing import StreamMode
from slet_sdk.aelite.utils.device_info import get_device_string

if TYPE_CHECKING:
    from slet_sdk.aelite.manifest import AgentManifest
    from slet_sdk.aelite.resources.resource import AeliteResource


# ---------------------------------------------------------------------------
# Внутренние маркеры для очередей
# ---------------------------------------------------------------------------


@dataclass
class _End:
    """Сигнал завершения потока данных в очереди."""

    pass


@dataclass
class _Error:
    """Сигнал ошибки в очереди."""

    error: ErrorResponse


# Тип элемента очереди: либо текстовый чанк, либо служебный маркер.
_QueueItem = str | _End | _Error


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------


class AgentSession:
    """
    Сессия общения с агентом через WebSocket.

    Жизненный цикл
    --------------
    1. Создайте экземпляр через ``client.aelite.deploy_and_connect()``.
    2. Вызовите ``await session.connect()`` — соединение устанавливается,
       фоновый listener запускается автоматически.
    3. Используйте ``chat()`` / ``stream()`` для диалога.
    4. Зарегистрируйте коллбэки (``on_message``, ``on_error`` и др.)
       для получения незапрошенных сообщений и уведомлений.
    5. Завершите работу через ``await session.disconnect()``.

    Особенности
    -----------
    - Режим «Запрос — Ответ»: ``chat()`` и ``stream()``.
    - Инициатива агента: незапрошенные сообщения маршрутизируются
      через ``on_message`` (буфер) или ``on_incoming_stream`` (стрим).
    - Автоматическая обработка вызовов клиентских инструментов.
    - Автоматический реконнект при разрыве соединения.
    """

    def __init__(
        self,
        *,
        # Main
        thread_id: str,
        headers: dict,
        device: str = get_device_string(),
        stream_mode: StreamMode | str = StreamMode.tokens,
        manifest: AgentManifest | None = None,
        resource: AeliteResource | None = None,
    ) -> None:
        self.thread_id = thread_id
        self.manifest = manifest
        self._resource = resource

        # Дочерние подагенты, доступные через session.SubAgentId
        self.sub: dict[str, AgentSession] = {}

        # URL-адреса
        ws_base = self._resource.base_ws_url
        base_url = self._resource.base_url
        self.device = device
        self.stream_mode = stream_mode
        self.http_url = base_url
        self.ws_url = ws_base
        self._ws_connect_url = (
            f"{self.ws_url}/agent/ws/{self.thread_id}"
            f"?device={self.device}"
            f"&stream_mode={self.stream_mode}"
        )

        self.headers = headers
        self.websocket = None
        self.logger = self._resource.logger
        self._is_connected = False

        self.logger.debug(f"Device name: {self.device}")
        self.logger.debug(f"Stream mode: {self.stream_mode}")

        # --- Настройки поведения ---

        # Если True: незапрошенные сообщения доставляются через on_incoming_stream
        # как асинхронный итератор (чанк за чанком).
        # Если False: текст накапливается в буфере и отдаётся целиком через on_message.
        self.stream_unsolicited: bool = False

        # --- Коллбэки ---
        self._add_callbacks_attributes()

        # --- Реестр клиентских инструментов ---
        self._registered_tools: dict[str, Callable] = {}

        # --- Внутренние механизмы ---

        self._listener_task: asyncio.Task | None = None

        # Очередь ответа на активный chat/stream запрос пользователя.
        self._response_queue: asyncio.Queue[_QueueItem] = asyncio.Queue()

        # Очередь стриминга незапрошенных сообщений (создаётся при необходимости).
        self._unsolicited_stream_queue: asyncio.Queue[_QueueItem] | None = None

        # Буфер текста незапрошенного сообщения (stream_unsolicited=False).
        self._unsolicited_buffer: list[str] = []

        # True, пока мы ожидаем ответ на явный запрос пользователя.
        self._waiting_for_response: bool = False

        # True, пока агент передаёт какие-либо данные (блокирует отправку новых запросов).
        self._is_incoming_traffic: bool = False

        # --- Настройки реконнекта ---
        self.max_reconnect_attempts: int = (
            10  # float("inf") для бесконечного реконнекта
        )
        self.reconnect_delay: int = 5  # секунд между попытками
        self._reconnect_attempt: int = 0  # текущий номер попытки (только для логов)
        self._manual_disconnect: bool = (
            False  # True = закрыто намеренно, реконнект не нужен
        )

        # Event сброшен (clear) на время реконнекта; chat/stream ждут его.
        self._reconnect_event: asyncio.Event = asyncio.Event()
        self._reconnect_event.set()

        # Мьютекс: запрещает параллельные вызовы chat/stream.
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Подключение / отключение
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Устанавливает WebSocket-соединение и запускает фоновый listener.

        Raises
        ------
        SletClientError
            Если сервер вернул ошибочный HTTP-статус при handshake.
        """
        try:
            await self._connect_websocket()
            self._is_connected = True
            self._listener_task = asyncio.create_task(self._listen_loop())
            self.logger.info(f"Connected to agent at {self.ws_url}")

        except InvalidStatus as e:
            self.logger.error(f"Failed to connect to agent: {e}")
            err = SletClientError(
                ErrorResponse(
                    status=e.response.status_code,
                    error=ErrorCode.WEBSOCKET_ERROR,
                    message=str(e),
                )
            )
            self._invoke_callback(self.on_error, err)
            raise err from e

        except Exception as e:
            self.logger.error(f"Unexpected connect error: {e}")
            self._invoke_callback(self.on_error, e)
            raise

    async def disconnect(self) -> None:
        """
        Останавливает фоновый listener и закрывает WebSocket-соединение.

        После вызова реконнект не выполняется.
        """
        self._manual_disconnect = True
        self._is_connected = False

        await self._cancel_listener()

        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from agent")

    async def disconnect_all(self) -> None:
        """Рекурсивно отключает всё дерево подагентов, затем текущую сессию."""
        for sub_session in self.sub.values():
            await sub_session.disconnect_all()
        await self.disconnect()

    async def redeploy(self, new_manifest: AgentManifest | None = None) -> None:
        """
        Пересоздаёт агента на сервере, сохраняя ``thread_id`` (историю диалога).

        Parameters
        ----------
        new_manifest:
            Новый манифест. Если не передан — используется сохранённый манифест
            текущей сессии.

        Raises
        ------
        RuntimeError
            Если сессия была создана без ссылки на ресурс.
        ValueError
            Если манифест не передан и не был сохранён при создании сессии.
        """
        if self._resource is None:
            raise RuntimeError(
                "Cannot redeploy: session was created without resource reference. "
                "Use client.aelite.deploy_and_connect() to get a redeployable session."
            )

        manifest = new_manifest or self.manifest
        if manifest is None:
            raise ValueError("No manifest provided and no stored manifest.")

        self.manifest = manifest
        await self._resource.deploy(manifest, thread_id=self.thread_id)

    # ------------------------------------------------------------------
    # Регистрация инструментов
    # ------------------------------------------------------------------

    def register_tools(self, tools: list[Callable]) -> None:
        """
        Регистрирует функции клиентских инструментов.

        Имя инструмента берётся из атрибута ``_tool_name`` (если задан),
        иначе из ``__name__`` функции.

        Parameters
        ----------
        tools:
            Список вызываемых объектов (sync или async).
        """
        for fn in tools:
            name: str = getattr(fn, "_tool_name", None) or fn.__name__
            self._registered_tools[name] = fn
            self.logger.debug(f"Registered client tool: {name}")

    def register_tools_from_registry(self) -> None:
        """Загружает все инструменты из глобального реестра SDK."""
        self._registered_tools.update(get_registered_tools())

    # ------------------------------------------------------------------
    # Низкоуровневая отправка
    # ------------------------------------------------------------------

    async def send(self, message: str) -> None:
        """
        Отправляет сырое сообщение в WebSocket.

        Если в данный момент идёт реконнект — ждёт его завершения.

        Parameters
        ----------
        message:
            Строка для отправки (текст или JSON).

        Raises
        ------
        SletClientError
            Если соединение не установлено.
        """
        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(NetworkError(message="WebSocket is not connected."))

        await self.websocket.send(message)

    # ------------------------------------------------------------------
    # Публичные методы запросов
    # ------------------------------------------------------------------

    async def stream(
        self,
        message: str,
        files: list[Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Отправляет сообщение агенту и возвращает асинхронный генератор чанков ответа.

        Если агент в данный момент передаёт незапрошенное сообщение,
        метод ждёт его завершения перед отправкой.
        Параллельные вызовы ``chat``/``stream`` выстраиваются в очередь через мьютекс.

        Parameters
        ----------
        message:
            Текст сообщения пользователя.
        files:
            Список файлов для прикрепления. Каждый файл может быть ``str``,
            ``bytes`` или file-like объектом.

        Yields
        ------
        str
            Текстовые чанки ответа агента.

        Raises
        ------
        SletClientError
            При ошибке соединения или если агент вернул ошибку.
        """
        # Загружаем файлы и оборачиваем сообщение в JSON при наличии вложений
        attachments: list[dict] = []
        if files:
            for f in files:
                res = await self.upload_file(f)
                attachments.append({"ref": f"upload:{res['file_id']}"})

            message = json.dumps(
                {
                    "type": "message",
                    "text": message,
                    "attachments": attachments,
                }
            )

        # Ожидаем завершения реконнекта (если идёт)
        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(
                NetworkError(message="Cannot send message: not connected.")
            )

        async with self._lock:
            # Ждём завершения текущего незапрошенного сообщения от агента,
            # чтобы не смешать чужой хвост с нашим ответом.
            while self._is_incoming_traffic:
                await asyncio.sleep(0.05)

            # Сбрасываем мусор из предыдущего цикла
            while not self._response_queue.empty():
                self._response_queue.get_nowait()

            self._waiting_for_response = True

            try:
                payload = {"type": "message", "text": message}
                if attachments:
                    payload["attachments"] = attachments
                await self.send(json.dumps(payload))

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
                self._waiting_for_response = False

    async def chat(
        self,
        message: str,
        files: list[Any] | None = None,
    ) -> str:
        """
        Отправляет сообщение и возвращает полный текстовый ответ агента.

        Удобная обёртка над ``stream()``, собирающая чанки в единую строку.

        Parameters
        ----------
        message:
            Текст сообщения пользователя.
        files:
            Список файлов для прикрепления.

        Returns
        -------
        str
            Полный ответ агента.
        """
        chunks: list[str] = []
        async for chunk in self.stream(message=message, files=files):
            chunks.append(chunk)
        return "".join(chunks)

    async def upload_file(self, file: Any) -> dict:
        """
        Загружает файл на сервер через HTTP.

        Parameters
        ----------
        file:
            - ``str``       — текстовое содержимое (text/plain, UTF-8).
            - ``bytes``     — сырые байты (application/octet-stream).
            - file-like     — объект с методом ``.read()``
              (``BufferedReader``, ``SpooledTemporaryFile`` и т.д.).

        Returns
        -------
        dict
            Ответ сервера, содержащий как минимум ``file_id``.

        Raises
        ------
        TypeError
            Если тип аргумента не поддерживается.
        """
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
            filename = Path(raw_name).name if raw_name else "upload.bin"
            mime_type, _ = mimetypes.guess_type(filename)
            mime_type = mime_type or "application/octet-stream"

            if asyncio.iscoroutinefunction(getattr(file, "read", None)):
                file_data = await file.read()
            else:
                file_data = file.read()

        else:
            raise TypeError(
                f"Unsupported file type: {type(file).__name__}. "
                "Expected str, bytes, or file-like object."
            )
        upload_url = "/upload/"
        return await self._resource._request(
            "POST",
            upload_url,
            files={"file": (filename, file_data, mime_type)},
            data={"mime_type": mime_type},
        )

    async def trigger(self, name: str, payload: dict[str, Any] | None = None) -> None:
        """
        Отправляет триггер (событие) агенту без ожидания ответа.

        Для получения реакции агента используйте колбэки
        ``on_message`` или ``on_incoming_stream``.

        Parameters
        ----------
        name:
            Имя триггера.
        payload:
            Произвольные данные события.

        Raises
        ------
        ConnectionError
            Если WebSocket не подключён.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected.")

        event = {"type": "trigger", "name": name, "payload": payload or {}}
        await self.send(json.dumps(event))

    async def set_stream_mode(self, stream_mode: StreamMode) -> None:
        """
        Меняет режим стриминга сообщений от агента.

        Не влияет на уже запущенные запросы к агенту,
        вступает в силу на следующем запросе.

        Parameters
        ----------
        stream_mode:
            Режим стриминга сообщений.

        Raises
        ------
        ConnectionError
            Если WebSocket не подключён.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected.")

        event = {"type": "set_stream_mode", "mode": stream_mode}
        await self.send(json.dumps(event))

    # ------------------------------------------------------------------
    # Вспомогательные приватные методы
    # ------------------------------------------------------------------

    async def _connect_websocket(self) -> None:
        self.websocket = await self._resource.websockets.connect(
            self._ws_connect_url,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            additional_headers=self.headers,
        )

    def _add_callbacks_attributes(self) -> None:
        """
        Добавляет атрибуты для коллбэков.
        Коллбэки поддерживают как async, так и sync функции.
        """

        # Вызывается при получении полного незапрошенного сообщения.
        # (stream_unsolicited=False)
        self.on_message: (Callable[[str], None | Awaitable[None]]) | None = None

        # Вызывается при начале незапрошенного потока.
        # (stream_unsolicited=True)
        self.on_incoming_stream: (
            Callable[[AsyncIterable[str]], None | Awaitable[None]]
        ) | None = None

        # Уведомление о старте инструмента на стороне сервера.
        self.on_tool_start: (Callable[[str], None | Awaitable[None]]) | None = None

        # Глобальный обработчик исключений.
        self.on_error: (Callable[[Exception], None | Awaitable[None]]) | None = None

        # Глобальный обработчик сообщений об ошибках от сервера.
        self.on_server_error: (
            Callable[[ErrorResponse], None | Awaitable[None]]
        ) | None = None

        # Уведомление о смене модели / провайдера.
        self.on_model_change: (Callable[[str, str], None | Awaitable[None]]) | None = (
            None
        )

    async def _cancel_listener(self) -> None:
        """
        Отменяет фоновый listener и дожидается его завершения.

        Безопасен для вызова, даже если listener уже завершён или не запускался.
        """
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        self._listener_task = None

    async def _restart_listener(self) -> None:
        """
        Гарантированно останавливает старый listener и запускает новый.

        Вызывается после успешного реконнекта, чтобы исключить ситуацию,
        когда два listener одновременно читают один и тот же сокет.
        """
        await self._cancel_listener()
        self._listener_task = asyncio.create_task(self._listen_loop())

    def _cleanup_queues(self) -> None:
        """
        Разблокирует всех ожидающих при разрыве соединения и очищает буфера.

        Должна вызываться из ``_listen_loop`` перед попыткой реконнекта.
        """
        # Сбрасываем флаг явно, чтобы исключить гонку между _cleanup_queues
        # и finally-блоком stream(): если новые токены придут после реконнекта
        # до выхода из stream(), они не попадут в «старую» очередь.
        self._waiting_for_response = False
        self._is_incoming_traffic = False

        # Разблокируем stream(), если он висит на get()
        self._response_queue.put_nowait(_End())

        # Завершаем незапрошенный стрим, если был активен
        if self._unsolicited_stream_queue:
            self._unsolicited_stream_queue.put_nowait(_End())
            self._unsolicited_stream_queue = None

        self._unsolicited_buffer.clear()

    # ------------------------------------------------------------------
    # Фоновый цикл чтения WebSocket
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """
        Постоянно читает WebSocket и маршрутизирует входящие данные.

        Маршрутизация
        -------------
        - Системные события (JSON с полем ``type``) → соответствующие обработчики.
        - Текстовые токены во время активного chat/stream → ``_response_queue``.
        - Текстовые токены при инициативе агента → буфер или
          ``_unsolicited_stream_queue`` (зависит от ``stream_unsolicited``).

        При разрыве соединения запускает ``_reconnect_loop``,
        если ``disconnect()`` не был вызван вручную.
        """
        self.logger.debug("Background listener started.")
        try:
            async for raw_msg in self.websocket:
                is_system_msg = False

                # Оптимизация: парсим JSON только если сообщение похоже на объект
                if raw_msg.startswith("{"):
                    try:
                        data = json.loads(raw_msg)
                        if isinstance(data, dict) and "type" in data:
                            is_system_msg = True
                            await self._dispatch_event(data)
                            continue
                    except json.JSONDecodeError:
                        # Начинается с '{', но не является JSON — обрабатываем как текст
                        pass

                # --- Текстовый токен ---
                if not is_system_msg:
                    if self._waiting_for_response:
                        # Режим диалога: кладём токен в очередь ответа
                        await self._response_queue.put(raw_msg)
                    else:
                        # Инициатива агента: поднимаем флаг, чтобы chat() не вклинился
                        self._is_incoming_traffic = True
                        await self._handle_unsolicited_token(raw_msg)

        except ConnectionClosed as e:
            self.logger.warning(f"WebSocket closed: {e}")
            await self._on_listener_failure(
                NetworkError(message=f"WebSocket connection closed: {e}")
            )

        except Exception as e:
            self.logger.error(f"Error in background listener: {e}")
            await self._on_listener_failure(
                NetworkError(message=f"Listener crashed: {e}")
            )

    async def _dispatch_event(self, data: dict) -> None:
        """
        Обрабатывает системное WebSocket-событие (JSON с полем ``type``).

        Parameters
        ----------
        data:
            Распарсенный JSON-объект события.
        """
        event_type: str = data.get("type", "")

        # --- Конец ответа ---
        if event_type == "end_of_answer":
            self._is_incoming_traffic = False

            if self._waiting_for_response:
                # Завершаем активный stream()
                await self._response_queue.put(_End())
            else:
                # Конец незапрошенного сообщения
                if self.stream_unsolicited:
                    if self._unsolicited_stream_queue:
                        await self._unsolicited_stream_queue.put(_End())
                        self._unsolicited_stream_queue = None
                else:
                    if self._unsolicited_buffer:
                        full_text = "".join(self._unsolicited_buffer)
                        self._unsolicited_buffer.clear()
                        self._invoke_callback(self.on_message, full_text)

        # --- Вызов клиентского инструмента ---
        elif event_type == "client_tool_call":
            asyncio.create_task(self._handle_client_tool_call(data.get("payload", {})))

        # --- Уведомление о старте инструмента ---
        elif event_type == "tool_start":
            name: str = data.get("payload", {}).get("name", "")
            self._invoke_callback(self.on_tool_start, name)

        # --- Смена модели / провайдера ---
        elif event_type == "model_changed":
            payload = data.get("payload", {})
            self._invoke_callback(
                self.on_model_change,
                payload.get("provider"),
                payload.get("model"),
            )

        # --- Ошибка от сервера ---
        elif event_type == "error":
            self._is_incoming_traffic = False

            # Закрываем незавершённые стримы/буфера
            if self._unsolicited_stream_queue:
                await self._unsolicited_stream_queue.put(_End())
                self._unsolicited_stream_queue = None
            self._unsolicited_buffer.clear()

            err = ErrorResponse(**data.get("payload", {}))

            if self._waiting_for_response:
                await self._response_queue.put(_Error(err))

            self._invoke_callback(self.on_server_error, err)

    async def _handle_unsolicited_token(self, token: str) -> None:
        """
        Маршрутизирует токен незапрошенного сообщения в стрим или буфер.

        Parameters
        ----------
        token:
            Текстовый чанк от агента.
        """
        if self.stream_unsolicited:
            if self._unsolicited_stream_queue is None:
                # Первый токен новой волны — создаём очередь и уведомляем пользователя
                self._unsolicited_stream_queue = asyncio.Queue()
                gen = self._make_unsolicited_generator(self._unsolicited_stream_queue)
                self._invoke_callback(self.on_incoming_stream, gen)

            await self._unsolicited_stream_queue.put(token)
        else:
            self._unsolicited_buffer.append(token)

    async def _on_listener_failure(self, error: NetworkError) -> None:
        """
        Вызывается при неожиданном завершении ``_listen_loop``.

        Сбрасывает состояние, вызывает ``on_error`` и запускает реконнект,
        если соединение не было закрыто вручную.

        Parameters
        ----------
        error:
            Описание причины сбоя.
        """
        self._is_connected = False
        self._cleanup_queues()

        if not self._manual_disconnect:
            if self.on_error:
                self._invoke_callback(self.on_error, SletClientError(error))
            # Запускаем реконнект как отдельную задачу.
            # Используем add_done_callback, чтобы не потерять исключение молча.
            task = asyncio.create_task(self._reconnect_loop())
            task.add_done_callback(self._on_reconnect_task_done)

    def _on_reconnect_task_done(self, task: asyncio.Task) -> None:
        """
        Колбэк завершения задачи реконнекта.

        Вытаскивает исключение из задачи, чтобы оно не «проглотилось» молча.
        """
        if not task.cancelled():
            exc = task.exception()
            if exc:
                self.logger.error(f"Reconnect loop failed with exception: {exc}")

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _invoke_callback(self, callback: Callable | None, *args: Any) -> None:
        """
        Безопасно вызывает колбэк в текущем event loop.

        - Async-функции планируются через ``asyncio.create_task()``.
        - Sync-функции вызываются напрямую.

        .. warning::
            Синхронные колбэки выполняются в потоке event loop.
            Блокирующие операции в них затормозят ``_listen_loop``.
            По возможности используйте async-колбэки.

        Parameters
        ----------
        callback:
            Вызываемый объект или ``None``.
        *args:
            Аргументы для колбэка.
        """
        if not callback:
            return

        try:
            if inspect.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args))
            else:
                self.logger.debug(
                    f"{callback.__name__} is a sync callback. "
                    "Make sure it does not perform any blocking operations."
                )
                callback(*args)

        except Exception as e:
            self.logger.error(f"Error invoking callback {callback}: {e}")

            # Избегаем рекурсии: не вызываем on_error из самого on_error
            if self.on_error is not None and callback is not self.on_error:
                try:
                    cb_error = SletClientError(
                        CallbackError(message=str(e)),
                    )
                    if inspect.iscoroutinefunction(self.on_error):
                        asyncio.create_task(self.on_error(cb_error))
                    else:
                        self.on_error(cb_error)
                except Exception as inner_e:
                    self.logger.error(f"Error in on_error callback: {inner_e}")

    async def _make_unsolicited_generator(
        self,
        queue: asyncio.Queue[_QueueItem],
    ) -> AsyncGenerator[str, None]:
        """
        Генератор, читающий чанки незапрошенного сообщения из очереди.

        Завершается при получении ``_End`` или бросает ``SletClientError``
        при ``_Error``.

        Parameters
        ----------
        queue:
            Очередь, в которую ``_listen_loop`` кладёт токены.

        Yields
        ------
        str
            Текстовые чанки.
        """
        while True:
            item = await queue.get()
            match item:
                case str():
                    yield item
                case _End():
                    break
                case _Error(error):
                    raise SletClientError(error)

    async def _handle_client_tool_call(self, payload: dict) -> None:
        """
        Обрабатывает вызов клиентского инструмента, инициированный агентом.

        Ищет зарегистрированный инструмент по имени, вызывает его
        и отправляет результат обратно на сервер.

        Parameters
        ----------
        payload:
            Словарь с полями ``name``, ``args`` и ``tool_call_id``.
        """
        tool_name: str = payload.get("name", "")
        args: dict = payload.get("args", {})
        call_id: str = payload.get("tool_call_id", "")

        self.logger.debug(f"Handling tool call: {tool_name}")

        fn = self._registered_tools.get(tool_name)
        result: str

        if not fn:
            result = f"Error: Tool '{tool_name}' not registered on client."
            self.logger.warning(result)
        else:
            try:
                if inspect.iscoroutinefunction(fn):
                    result = str(await fn(**args))
                else:
                    loop = asyncio.get_running_loop()
                    result = str(await loop.run_in_executor(None, lambda: fn(**args)))
            except Exception as e:
                result = f"Error executing tool '{tool_name}': {e}"
                self.logger.error(result)
                self._invoke_callback(self.on_error, e)

        response = {
            "type": "client_tool_result",
            "payload": {
                "tool_call_id": call_id,
                "result": result,
            },
        }

        if self._is_connected:
            await self.websocket.send(json.dumps(response))

    # ------------------------------------------------------------------
    # Реконнект
    # ------------------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        """
        Пытается восстановить WebSocket-соединение после разрыва.

        Выполняет до ``max_reconnect_attempts`` попыток с паузой
        ``reconnect_delay`` секунд между ними.

        При каждой неудаче вызывает ``on_error``.
        После успешного реконнекта перезапускает ``_listen_loop``.

        Raises
        ------
        SletClientError
            Если все попытки исчерпаны.
        """
        self._reconnect_attempt = 0
        self._reconnect_event.clear()  # блокируем chat/stream на время реконнекта

        while self._reconnect_attempt < self.max_reconnect_attempts:
            self._reconnect_attempt += 1
            is_last = self._reconnect_attempt == self.max_reconnect_attempts

            self.logger.info(
                f"Reconnect attempt {self._reconnect_attempt}/{self.max_reconnect_attempts}"
            )

            try:
                await self._connect_websocket()

                self._is_connected = True
                self._manual_disconnect = False

                # Гарантированно останавливаем старый listener (если вдруг ещё жив)
                # и запускаем новый. Без этого два listener могли бы одновременно
                # читать один сокет и перемешивать сообщения.
                await self._restart_listener()

                self._reconnect_event.set()  # разблокируем chat/stream
                self.logger.info("WebSocket successfully reconnected.")
                return  # ✅ Успех

            except Exception as e:
                message = (
                    "WebSocket reconnection failed. Maximum attempts reached."
                    if is_last
                    else (
                        f"WebSocket reconnection attempt "
                        f"{self._reconnect_attempt} failed: {e}"
                    )
                )
                self.logger.error(message)

                error = NetworkError(message=f"{message}: {e}")
                if self.on_error:
                    self._invoke_callback(self.on_error, SletClientError(error))

                if is_last:
                    # Разблокируем ожидающих, чтобы они не зависли навсегда
                    self._reconnect_event.set()
                    raise SletClientError(error)

                await asyncio.sleep(self.reconnect_delay)

    # ------------------------------------------------------------------
    # Magic: доступ к подагентам через атрибуты
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> AgentSession:
        """
        Магический доступ к дочерним подагентам через атрибуты экземпляра.

        Пример использования::

            session.SubAgentId.connect()
            await session.SubAgentId.chat("hello")

        Parameters
        ----------
        name:
            Идентификатор подагента.

        Raises
        ------
        AttributeError
            Если подагент с таким именем не зарегистрирован.
        """
        # Приватные и dunder-атрибуты пробрасываем стандартно, чтобы избежать рекурсии
        if name.startswith("_"):
            raise AttributeError(name)

        if name in self.sub:
            return self.sub[name]

        raise AttributeError(
            f"Sub-agent '{name}' not found. "
            f"Available: {list(self.sub.keys()) or 'none registered'}."
        )
