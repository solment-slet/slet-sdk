import json
import asyncio
import inspect
import logging
from typing import AsyncGenerator, AsyncIterable, Callable, Optional, Dict, Any, List, Union, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

from slet_sdk.schemas import ErrorResponse, ErrorCode
from slet_sdk.schemas.errors import CallbackError, BackgroundListenerError
from slet_sdk.exceptions import SletClientError

logger = logging.getLogger(__name__)

# Специальный объект-маркер для очереди, обозначающий конец потока данных
_SENTINEL = object()
# Специальный объект-маркер для очереди, обозначающий ошибку при получении ответа от сервера
class _ErrorSentinel:
    def __init__(self, error: ErrorResponse):
        self.error = error


class AgentSession:
    """
    Сессия общения с агентом через WebSocket.

    Особенности:
    - Фоновая задача (listener) запускается автоматически при вызове connect().
    - Поддерживает режим "Запрос-Ответ" (chat, stream).
    - Поддерживает инициативу агента (незапрошенные сообщения).
    - Автоматически обрабатывает вызовы инструментов (client tools).
    """

    def __init__(self, ws_url: str, headers: dict):
        self.ws_url = ws_url
        self.headers = headers
        self.websocket = None
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
        self.on_model_change: Optional[Callable[[str], Union[None, Awaitable[None]]]] = None

        # Реестр инструментов
        self._registered_tools: Dict[str, Callable] = {}

        # --- Внутренние механизмы ---
        self._listener_task: Optional[asyncio.Task] = None

        # Очередь для ответов на активный запрос пользователя (chat/stream)
        self._response_queue: asyncio.Queue = asyncio.Queue()

        # Очередь для стриминга незапрошенных сообщений (если stream_unsolicited=True)
        self._unsolicited_stream_queue: Optional[asyncio.Queue] = None

        # Буфер для накопления текста незапрошенного сообщения (если stream_unsolicited=False)
        self._unsolicited_buffer: List[str] = []

        # Флаг: мы ждем ответ на явный запрос пользователя?
        self._waiting_for_response = False

        # Флаг: прямо сейчас идет прием данных от агента (любых)
        # Нужен для блокировки отправки новых сообщений, пока агент не закончит текущую мысль.
        self._is_incoming_traffic = False

        # Блокировка, чтобы нельзя было вызвать chat/stream параллельно
        self._lock = asyncio.Lock()

    async def connect(self):
        """Устанавливает соединение и запускает фоновый слушатель."""
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
                additional_headers=self.headers,
            )
            self._is_connected = True

            # АВТОМАТИЧЕСКИЙ ЗАПУСК ФОНОВОЙ ЗАДАЧИ
            self._listener_task = asyncio.create_task(self._listen_loop())

            logger.info(f"Connected to agent at {self.ws_url}")
        except websockets.exceptions.InvalidStatus as e:
            logger.error(f"Failed to connect to agent: {e}")

            err = ErrorResponse(
                status=e.response.status_code,
                error=ErrorCode.WEBSOCKET_ERROR,
                message=str(e),
            )
            self._invoke_callback(self.on_error, err)
            raise SletClientError(err)

    async def disconnect(self):
        """Останавливает фоновые задачи и закрывает сокет."""
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
            logger.info("Disconnected from agent")

    def register_tools(self, tools: List[Callable]):
        """Регистрирует функции инструментов."""
        for fn in tools:
            name = getattr(fn, '_tool_name', None) or fn.__name__
            self._registered_tools[name] = fn
            logger.debug(f"Registered client tool: {name}")

    def register_tools_from_registry(self):
        """Загружает инструменты из глобального реестра SDK (если есть)."""
        try:
            from slet_sdk.tools import get_registered_tools
            self._registered_tools.update(get_registered_tools())
        except ImportError:
            pass

    async def send(self, message: str):
        """Низкоуровневая отправка сообщения в сокет."""
        if not self._is_connected:
            raise ConnectionError("Agent session is not connected")
        await self.websocket.send(message)

    # ------------------------------------------------------------------
    # Публичные методы запросов
    # ------------------------------------------------------------------

    async def stream(self, message: str) -> AsyncGenerator[str, None]:
        """
        Отправляет сообщение агенту и возвращает генератор с ответом.
        Блокируется, если агент в данный момент уже передает какое-то сообщение.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected")

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
                    chunk = await self._response_queue.get()

                    if chunk is _SENTINEL:
                        break  # Конец ответа
                    elif isinstance(chunk, _ErrorSentinel):
                        raise SletClientError(chunk.error)

                    yield chunk

            finally:
                # Выключаем режим ожидания
                self._waiting_for_response = False

    async def chat(self, message: str) -> str:
        """
        Отправляет сообщение и возвращает полный текстовый ответ.
        Работает через stream(), собирая чанки в строку.
        """
        full_response = []
        async for chunk in self.stream(message):
            full_response.append(chunk)
        return "".join(full_response)

    async def upload_file(self, filepath: str) -> dict:
        """
        Загружает файл на сервер через HTTP.
        Возвращает словарь с file_id.
        """
        import aiohttp
        import mimetypes

        # Определяем MIME
        mime, _ = mimetypes.guess_type(filepath)
        mime = mime or "application/octet-stream"
        filename = filepath.split("/")[-1].split("\\")[-1]

        # Строим HTTP URL на основе WS URL
        # ws://host/ae/agent/ws/{id} -> http://host/ae/upload/
        http_base = self.ws_url.split("/ae/agent/ws/")[0]
        http_base = http_base.replace("ws://", "http://").replace("wss://", "https://")
        upload_url = f"{http_base}/ae/upload/"

        form = aiohttp.FormData()
        # Открываем файл
        try:
            file_obj = open(filepath, "rb")
        except OSError as e:
            raise RuntimeError(f"Failed to open file {filepath}: {e}")

        form.add_field("file", file_obj, filename=filename, content_type=mime)
        form.add_field("mime_type", mime)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(upload_url, data=form) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"Upload failed: {resp.status} - {text}")
                    return await resp.json()
        finally:
            file_obj.close()

    async def send_with_files(self, text: str, files: list[str]) -> AsyncGenerator[str, None]:
        """
        Загружает файлы, отправляет сообщение и стримит ответ.

        :param text: Текст сообщения
        :param files: Список путей к файлам
        :return: Генератор токенов ответа
        """
        if not self._is_connected:
            raise ConnectionError("Not connected")

        # 1. Загружаем файлы
        attachments = []
        for fpath in files:
            res = await self.upload_file(fpath)
            attachments.append({"ref": f"upload:{res['file_id']}"})

        # 2. Формируем payload
        payload = {
            "type": "message",
            "text": text,
            "attachments": attachments
        }
        message_json = json.dumps(payload)

        # 3. Отправляем через существующую логику stream()
        async for chunk in self.stream(message_json):
            yield chunk

    async def trigger(self, name: str, payload: Dict[str, Any] = None) -> Any:
        """
        Отправляет триггер (событие) агенту и ждет ответ.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected")

        event = {"type": "trigger", "name": name, "payload": payload or {}}

        async with self._lock:
            # Очистка очереди
            while not self._response_queue.empty():
                self._response_queue.get_nowait()

            # Ждем освобождения канала
            while self._is_incoming_traffic:
                await asyncio.sleep(0.05)

            self._waiting_for_response = True
            try:
                await self.websocket.send(json.dumps(event))

                full_response = []
                while True:
                    chunk = await self._response_queue.get()
                    if chunk is _SENTINEL:
                        break
                    full_response.append(chunk)
                return "".join(full_response)
            finally:
                self._waiting_for_response = False

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
        logger.debug("Background listener started")
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
                                    await self._response_queue.put(_SENTINEL)

                                # Сценарий Б: Это конец незапрошенного сообщения
                                else:
                                    if self.stream_unsolicited:
                                        # Завершаем стрим
                                        if self._unsolicited_stream_queue:
                                            await self._unsolicited_stream_queue.put(_SENTINEL)
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
                                if self._waiting_for_response:
                                    # Если кто-то ожидает ответ, прокидываем ему ошибку
                                    await self._response_queue.put(_ErrorSentinel(
                                        error=ErrorResponse(**data.get("payload")
                                    )))
                                self._invoke_callback(self.on_error, data.get("payload"))
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

        except ConnectionClosed:
            logger.info("WebSocket closed in background listener")
            self._is_connected = False
            self._cleanup_queues()

        except Exception as e:
            logger.error(f"Error in background listener: {e}")
            self._invoke_callback(self.on_error, BackgroundListenerError(message=str(e)))
            self._cleanup_queues()

    def _cleanup_queues(self):
        """Освобождает всех ожидающих при разрыве соединения."""
        if self._waiting_for_response:
            self._response_queue.put_nowait(_SENTINEL)
        if self._unsolicited_stream_queue:
            self._unsolicited_stream_queue.put_nowait(_SENTINEL)

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
                logger.warning(f"{callback.__name__} is not a coroutine function. Make sure that it does not perform any blocking actions.")
                callback(*args)
        except Exception as e:
            logger.error(f"Error invoking callback {callback}: {e}")
            # Вызов on_error, если это не он сам
            if self.on_error is not None and callback is not self.on_error:
                try:
                    if inspect.iscoroutinefunction(self.on_error):
                        asyncio.create_task(self.on_error(CallbackError(message=str(e))))
                    else:
                        self.on_error(CallbackError(message=str(e)))
                except Exception as err:
                    # Предотвращаем рекурсию или крэш
                    logger.error(f"Error in on_error callback: {err}")

    async def _make_unsolicited_generator(self, queue: asyncio.Queue) -> AsyncGenerator[str, None]:
        """Генератор, который читает из очереди незапрошенных сообщений."""
        while True:
            chunk = await queue.get()
            if chunk is _SENTINEL:
                break
            yield chunk

    async def _handle_client_tool_call(self, payload: dict):
        """Обработка вызова клиентского инструмента."""
        tool_name = payload.get("name", "")
        args = payload.get("args", {})
        call_id = payload.get("tool_call_id", "")

        logger.debug(f"Handling tool call: {tool_name}")
        fn = self._registered_tools.get(tool_name)

        result = None
        if not fn:
            result = f"Error: Tool '{tool_name}' not registered on client"
            logger.warning(result)
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
                logger.error(f"Tool execution failed: {e}")

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
