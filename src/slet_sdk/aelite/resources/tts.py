from slet_sdk.core.mixins import BaseResource
import asyncio
import json
import re
from typing import AsyncGenerator, AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.connection import Connection

from slet_sdk.aelite.manifest import TTSConfig

# URL для TTS WS
GATEWAY_URL = "ws://localhost:8000/ae/tts/ws"


class TTSResource(BaseResource):
    async def stream_tts(
        self,
        text_iterator: AsyncIterator[str],
        tts_config: TTSConfig = TTSConfig(),
        smart_buffering: bool = True,
        smart_buffer_max_chunk_size: int = 200,
    ) -> AsyncGenerator[bytes, None]:
        """
        Потоковый клиент для Piper TTS. Принимает поток текста (например, от LLM),
        автоматически буферизует его по предложениям и возвращает поток аудио-байтов.

        Args:
            text_iterator (AsyncIterator[str]): Асинхронный генератор, выдающий куски текста (токены).
            tts_config (TTSConfig): Настройки TTS.
            smart_buffering (bool): Если True, собирает токены в предложения перед отправкой.
            smart_buffer_max_chunk_size (int): Лимит буфера (символов), чтобы не копить бесконечно, если нет точек.

        Yields:
            bytes: Чанки аудио (Raw PCM int16).

        Raises:
            ConnectionClosed: Если сервер разорвал соединение.
            Exception: При ошибках внутри WebSocket.
        """
        url = f"{self.base_ws_url}"

        # Конфигурация сессии
        session_config = tts_config.model_dump()
        # Очищаем None значения, чтобы не слать мусор
        session_config = {k: v for k, v in session_config.items() if v is not None}

        async with websockets.connect(url) as ws:
            # 1. Отправляем стартовый конфиг (можно и не отправлять, если устраивает дефолт,
            # но лучше задать голос явно)
            await ws.send(json.dumps(session_config))

            # Очередь для сигнализации об окончании отправки текста.
            # Используем Event, чтобы главный цикл знал, когда мы закончили слать текст
            sender_task_done = asyncio.Event()

            # ------------------------------------------------------------------
            # Внутренняя задача: Чтение из итератора -> Буферизация -> Отправка
            # ------------------------------------------------------------------
            async def _sender_loop():
                try:
                    if smart_buffering:
                        await self._smart_buffer_sender(
                            text_iterator, ws, smart_buffer_max_chunk_size
                        )
                    else:
                        await self._passthrough_sender(text_iterator, ws)
                except Exception as e:
                    self.logger.error(f"Ошибка в sender_loop: {e}")
                finally:
                    sender_task_done.set()

            # Запускаем отправку в фоне, чтобы не блокировать получение аудио
            sender_task = asyncio.create_task(_sender_loop())

            # ------------------------------------------------------------------
            # Главный цикл: Получение аудио и событий от сервера
            # ------------------------------------------------------------------
            try:
                while True:
                    try:
                        # Ждем сообщения от сервера.
                        # Если sender закончил работу, мы все равно продолжаем читать,
                        # пока сервер не пришлет {"event": "done"}
                        msg = await ws.recv()

                        if isinstance(msg, bytes):
                            # Это аудио-чанк
                            yield msg
                        else:
                            # Это JSON событие
                            event = json.loads(msg)

                            if event.get("event") == "done":
                                # Сервер подтвердил, что синтез полностью завершен
                                break

                            if event.get("event") == "error":
                                self.logger.error(
                                    f"TTS Server Error: {event.get('message')}"
                                )
                                # Можно рейзить ошибку или просто прерывать
                                break

                    except ConnectionClosed:
                        self.logger.warning("Соединение с TTS закрыто сервером.")
                        break

            finally:
                # Убедимся, что задача отправки тоже завершена (или отменяем её)
                if not sender_task.done():
                    sender_task.cancel()
                await sender_task

    # --------------------------------------------------------------------------
    # Логика буферизации
    # --------------------------------------------------------------------------

    async def _smart_buffer_sender(
        self,
        text_iter: AsyncIterator[str],
        ws: Connection,
        max_buffer_size: int,
    ):
        """
        Накапливает текст до знаков препинания (. ? ! \n) или до лимита длины,
        чтобы отправлять в TTS осмысленные фразы.
        """
        buffer = ""
        # Регулярка для поиска конца предложения: точка, вопрос, восклицание или перевод строки,
        # за которыми следует пробел или конец строки.
        # Группируем разделитель, чтобы оставить его в отправляемом куске.
        split_pattern = re.compile(r"([.?!]+(?:\s|$)|[\n]+)")

        async for chunk in text_iter:
            buffer += chunk

            while True:
                # Пытаемся найти разделитель предложения
                match = split_pattern.search(buffer)

                if match:
                    # Нашли конец предложения
                    split_idx = match.end()
                    sentence = buffer[:split_idx]
                    buffer = buffer[split_idx:]  # Остаток оставляем в буфере

                    # Отправляем готовое предложение
                    if sentence.strip():
                        await ws.send(json.dumps({"text": sentence}))

                elif len(buffer) > max_buffer_size:
                    # Буфер переполнен, а точки нет. Ищем хотя бы пробел, чтобы не резать слово.
                    last_space = buffer.rfind(" ")
                    if last_space != -1:
                        part = buffer[:last_space]
                        buffer = buffer[last_space:]  # Пробел и остаток оставляем
                        await ws.send(json.dumps({"text": part}))
                    else:
                        # Даже пробелов нет (очень длинное слово?), отправляем как есть
                        await ws.send(json.dumps({"text": buffer}))
                        buffer = ""
                    break  # Ждем следующих чанков

                else:
                    # Разделителей нет и буфер не полон -> ждем данных
                    break

        # Итератор закончился. Отправляем всё, что осталось в буфере с флагом last=True
        payload = {"text": buffer, "last": True}
        await ws.send(json.dumps(payload))

    async def _passthrough_sender(
        self,
        text_iter: AsyncIterator[str],
        ws: Connection,
    ):
        """
        Простой режим: отправляет чанки сразу, как они приходят.
        """
        async for chunk in text_iter:
            if chunk:
                await ws.send(json.dumps({"text": chunk}))

        # Сигнализируем о конце потока
        await ws.send(json.dumps({"last": True}))
