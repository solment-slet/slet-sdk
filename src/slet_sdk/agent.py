# Добавляем импорты
import json
import logging
from typing import AsyncGenerator, Callable, Optional, Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed

# Импортируем твой базовый клиент (предполагаем, что он в slet_sdk.client)
# from slet_sdk.client import SletClient

logger = logging.getLogger(__name__)

# =========================================================
# AGENT SESSION (WEBSOCKET WRAPPER)
# =========================================================


class AgentSession:
    """
    Сессия общения с агентом через WebSocket.
    Позволяет отправлять сообщения и слушать стрим ответов.
    """

    def __init__(self, ws_url: str, headers: dict):
        self.ws_url = ws_url
        self.headers = headers
        self.websocket = None
        self._is_connected = False

        # Колбэки для событий (опционально)
        self.on_token: Optional[Callable[[str], None]] = None
        self.on_tool_start: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

    async def connect(self):
        """Устанавливает WebSocket соединение с Keep-Alive."""
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                # extra_headers=self.headers, # Раскомментируй, если используешь
                # === KEEP ALIVE SETTINGS ===
                ping_interval=20,  # Отправлять пинг каждые N секунд
                ping_timeout=20,  # Ждать понг N секунд, иначе разрыв
                close_timeout=10,
            )
            self._is_connected = True
            logger.info(f"Connected to agent at {self.ws_url}")
        except Exception as e:
            logger.error(f"Failed to connect to agent: {e}")
            raise

    async def disconnect(self):
        """Закрывает соединение."""
        if self.websocket:
            await self.websocket.close()
            self._is_connected = False
            logger.info("Disconnected from agent")

    async def send(self, message: str):
        """Отправляет текстовое сообщение агенту."""
        if not self._is_connected:
            raise ConnectionError("Agent session is not connected")
        await self.websocket.send(message)

    async def trigger(
        self, name: str, payload: Dict[str, Any] = None, stream: bool = False
    ) -> Any:
        """
        Отправляет триггер и возвращает реакцию агента.
        Если stream=True, возвращает генератор токенов.
        Если stream=False, возвращает полный текст ответа.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected")

        event = {"type": "trigger", "name": name, "payload": payload or {}}
        await self.websocket.send(json.dumps(event))

        if stream:
            return self._listen_until_end()
        else:
            full_response = []
            async for chunk in self._listen_until_end():
                full_response.append(chunk)
            return "".join(full_response)

    async def chat(self, message: str) -> str:
        """
        Non-stream метод чата.
        1. Отправляет сообщение.
        2. Ждет ответа от агента.
        3. Собирает токены в строку.
        4. Возвращает полный ответ, когда агент закончил генерацию.
        """
        if not self._is_connected:
            raise ConnectionError("Agent session is not connected")

        await self.send(message)

        full_response = []

        # Используем внутренний генератор, который слушает до "end_of_answer"
        async for chunk in self._listen_until_end():
            full_response.append(chunk)

        return "".join(full_response)

    async def stream(self, message: str) -> AsyncGenerator[str, None]:
        """
        Stream метод чата.
        Отправляет сообщение и yield'ит токены по мере поступления,
        пока агент не закончит отвечать.
        """
        if not self._is_connected:
            raise ConnectionError("Agent session is not connected")

        await self.send(message)

        async for chunk in self._listen_until_end():
            yield chunk

    async def _listen_until_end(self) -> AsyncGenerator[str, None]:
        """
        Внутренний метод: слушает WebSocket до получения события 'end_of_answer'.
        """
        try:
            async for raw_msg in self.websocket:
                # Пытаемся распарсить JSON-событие
                try:
                    data = json.loads(raw_msg)
                    if isinstance(data, dict):
                        event_type = data.get("type")

                        # === СИГНАЛ ЗАВЕРШЕНИЯ ===
                        if event_type == "end_of_answer":
                            return  # Выходим из генератора (StopIteration)

                        # Обработка тулов (опционально)
                        if event_type == "tool_start":
                            if self.on_tool_start:
                                self.on_tool_start(data.get("payload", {}).get("name"))
                            continue

                except json.JSONDecodeError:
                    pass

                # Если это не JSON событие, значит это текст от LLM
                # (В твоем сервере send_message шлет plain text)
                yield raw_msg

        except ConnectionClosed:
            logger.info("WebSocket closed during stream")
            return

    async def listen_forever(self) -> AsyncGenerator[str, None]:
        """
        Асинхронный генератор, который слушает входящие сообщения.
        Возвращает текстовые чанки (токены) от LLM.
        Служебные события (Tool Call) обрабатывает сам или вызывает колбэки.
        """
        if not self._is_connected:
            raise ConnectionError("Agent session is not connected")

        try:
            async for message in self.websocket:
                # Пытаемся понять, это JSON-событие или просто текст
                try:
                    data = json.loads(message)

                    # Обработка событий
                    if isinstance(data, dict) and "type" in data:
                        event_type = data["type"]

                        if event_type == "tool_start":
                            tool_name = data.get("payload", {}).get("name", "unknown")
                            if self.on_tool_start:
                                self.on_tool_start(tool_name)
                            # Можно не yeild'ить это наружу, а просто логировать
                            continue

                        # Другие типы событий...
                except json.JSONDecodeError:
                    # Это обычный текстовый чанк от LLM
                    pass

                # Если это не JSON, или мы решили пропустить JSON дальше
                # yield message

                # В твоем сервере сейчас стриминг идет как текст.
                # Если сервер шлет JSON, то нужно менять логику.
                # Но пока сервер шлет plain text chunks для чата.

                # Проверка на старый формат "TOOL_CALL:"
                if isinstance(message, str) and message.startswith("TOOL_CALL:"):
                    tool_name = message.split(":")[1]
                    if self.on_tool_start:
                        self.on_tool_start(tool_name)
                    continue

                # Отдаем токен пользователю
                if self.on_token:
                    self.on_token(message)

                yield message

        except ConnectionClosed:
            logger.info("WebSocket connection closed")
            self._is_connected = False
        except Exception as e:
            logger.error(f"Error in listener: {e}")
            if self.on_error:
                self.on_error(str(e))
            raise
