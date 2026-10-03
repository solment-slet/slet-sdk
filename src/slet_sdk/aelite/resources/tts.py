from slet_sdk.core.mixins import BaseResource
import asyncio
import json
import re
from typing import AsyncGenerator, AsyncIterator

import websockets
from websockets.exceptions import ConnectionClosed
from websockets.asyncio.connection import Connection

from slet_sdk.aelite.manifest import PiperTTSConfig, SileroTTSConfig, TTSConfig
from slet_sdk.aelite.types import TTSStreamItem, TTSStreamConfig


class TTSResource(BaseResource):
    async def stream_tts(
        self,
        text_iterator: AsyncIterator[str],
        tts_config: TTSConfig | None = None,
        smart_buffering: bool = True,
        smart_buffer_max_chunk_size: int = 200,
    ) -> AsyncGenerator[TTSStreamItem, None]:
        """
        Streaming client for the TTS service (Piper or Silero). Consumes a
        stream of text (e.g. LLM tokens), buffers it into sentences
        automatically, and yields the resulting audio stream.

        The engine is selected by the type of `tts_config`:
        `PiperTTSConfig` (default) or `SileroTTSConfig`.

        Piper streams audio as it is generated. Silero does NOT support
        streaming: every text chunk sent to the server is synthesized
        completely before its audio is returned. The larger the chunk, the
        longer the delay before its first sound, so keep chunks small (smart
        buffering sends one sentence at a time, which is a good default).

        The first item yielded is always a `TTSStreamConfig` describing the
        audio format (sample_rate/sample_width/channels), sent by the server
        in its "config" event. It is produced when the first text chunk
        starts synthesizing (not at connection time), and is yielded again
        only if the format changes. Every other item is a `bytes` chunk of
        raw PCM int16 audio. Callers MUST check the type of each yielded
        item (e.g. `isinstance(item, TTSStreamConfig)`) before treating it
        as audio.

        Args:
            text_iterator (AsyncIterator[str]): Async generator producing
                chunks of text (tokens).
            tts_config (TTSConfig | None): Engine settings. None means
                `PiperTTSConfig()` (server default Piper voice).
            smart_buffering (bool): If True, accumulate tokens into full
                sentences before sending them to the server. Must be False
                for Silero with `ssml=True`, because splitting would break
                the SSML markup.
            smart_buffer_max_chunk_size (int): Buffer length limit (chars)
                so we don't accumulate forever when there's no punctuation.

        Yields:
            TTSStreamConfig: Before any audio, describing the audio format.
            bytes: Raw PCM int16 audio chunks.

        Raises:
            ValueError: If Silero SSML mode is combined with smart_buffering.
            ConnectionClosed: If the server closes the connection
                unexpectedly.
            RuntimeError: If the server reports an "error" event, or if
                audio arrives before a "config" event.
            Exception: On other internal WebSocket errors.
        """
        if tts_config is None:
            tts_config = PiperTTSConfig()

        # With SSML every text chunk must be a complete <speak>...</speak>
        # document. Sentence splitting would cut it into invalid fragments.
        if (
            isinstance(tts_config, SileroTTSConfig)
            and tts_config.ssml
            and smart_buffering
        ):
            raise ValueError(
                "smart_buffering must be disabled when ssml=True: each item of "
                "text_iterator has to be a complete SSML document."
            )

        url = f"{self.base_ws_url}/tts/ws"
        headers = {"Authorization": f"Bearer {self._access_token}"}

        # Session configuration. Includes "engine"; None values are dropped
        # so we don't send garbage to the server.
        session_config = tts_config.model_dump(exclude_none=True)

        async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
            # 1. Send the initial config (engine + engine parameters). The
            # server keeps these for the whole session. Invalid parameters
            # come back as an "error" event.
            await ws.send(json.dumps(session_config))

            # Event used to signal that we're done sending text.
            sender_task_done = asyncio.Event()

            # Last audio format announced by the server as
            # (sample_rate, sample_width, channels). None until the first
            # "config" event arrives.
            audio_format: tuple[int, int, int] | None = None

            # ------------------------------------------------------------------
            # Internal task: read from the iterator -> buffer -> send
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
                    self.logger.error(f"Error in sender_loop: {e}")
                finally:
                    sender_task_done.set()

            # Run the sender in the background so it doesn't block receiving
            # audio.
            sender_task = asyncio.create_task(_sender_loop())

            # ------------------------------------------------------------------
            # Main loop: receive audio and events from the server
            # ------------------------------------------------------------------
            try:
                while True:
                    try:
                        # Wait for a message from the server. Even if the
                        # sender has finished, we keep reading until the
                        # server sends {"event": "done"}.
                        msg = await ws.recv()

                        if isinstance(msg, bytes):
                            # This is an audio chunk.
                            if audio_format is None:
                                # Defensive: the server should always send
                                # "config" before any audio, but if it
                                # didn't, we can't tell the caller the
                                # audio format. Fail loudly instead of
                                # silently yielding un-describable bytes.
                                raise RuntimeError(
                                    "Received audio bytes before a 'config' "
                                    "event; cannot determine audio format."
                                )
                            yield msg
                        else:
                            # This is a JSON event.
                            event = json.loads(msg)
                            event_type = event.get("event")

                            if event_type == "config":
                                # Audio format descriptor. The server sends it
                                # before the first synthesis and again only if
                                # the format changes; extra fields (e.g.
                                # "engine") are ignored.
                                new_format = (
                                    event["sample_rate"],
                                    event["sample_width"],
                                    event["channels"],
                                )
                                if new_format != audio_format:
                                    audio_format = new_format
                                    yield TTSStreamConfig(
                                        sample_rate=new_format[0],
                                        sample_width=new_format[1],
                                        channels=new_format[2],
                                    )

                            elif event_type == "done":
                                # Server confirmed synthesis is fully done.
                                break

                            elif event_type == "error":
                                message = event.get("message")
                                self.logger.error(f"TTS Server Error: {message}")
                                raise RuntimeError(f"TTS Server Error: {message}")

                            elif event_type in ("synthesis_start", "synthesis_end"):
                                # Informational per-chunk markers; nothing
                                # for the caller to do with these right now.
                                self.logger.debug(
                                    f"TTS event: {event_type} "
                                    f"(index={event.get('index')})"
                                )

                            else:
                                self.logger.debug(
                                    f"Unhandled TTS event: {event_type}"
                                )

                    except ConnectionClosed:
                        self.logger.warning("TTS connection closed by server.")
                        break

            finally:
                # Make sure the sender task is also finished (or cancel it).
                # gather(return_exceptions=True) swallows the child's
                # CancelledError instead of re-raising it into the caller.
                if not sender_task.done():
                    sender_task.cancel()
                await asyncio.gather(sender_task, return_exceptions=True)

    # --------------------------------------------------------------------------
    # Buffering logic
    # --------------------------------------------------------------------------

    async def _smart_buffer_sender(
        self,
        text_iter: AsyncIterator[str],
        ws: Connection,
        max_buffer_size: int,
    ):
        """
        Accumulates text up to sentence-ending punctuation (. ? ! \\n) or up
        to a length limit, so we send meaningful phrases to the TTS server.
        """
        buffer = ""
        # Regex to find the end of a sentence: period, question mark,
        # exclamation mark, or newline, followed by whitespace or end of
        # string. The separator is captured so it stays in the sent chunk.
        split_pattern = re.compile(r"([.?!]+(?:\s|$)|[\n]+)")

        async for chunk in text_iter:
            buffer += chunk

            while True:
                # Try to find a sentence boundary.
                match = split_pattern.search(buffer)

                if match:
                    # Found the end of a sentence.
                    split_idx = match.end()
                    sentence = buffer[:split_idx]
                    buffer = buffer[split_idx:]  # Keep the remainder buffered.

                    # Send the completed sentence.
                    if sentence.strip():
                        await ws.send(json.dumps({"text": sentence}))

                elif len(buffer) > max_buffer_size:
                    # Buffer overflowed with no punctuation. Look for at
                    # least a space so we don't cut a word in half.
                    last_space = buffer.rfind(" ")
                    if last_space != -1:
                        part = buffer[:last_space]
                        buffer = buffer[last_space:]  # Keep space + remainder.
                        await ws.send(json.dumps({"text": part}))
                    else:
                        # No spaces at all (very long "word"?), send as-is.
                        await ws.send(json.dumps({"text": buffer}))
                        buffer = ""
                    break  # Wait for more chunks.

                else:
                    # No separators yet and buffer isn't full -> wait for
                    # more data.
                    break

        # Iterator exhausted. Send whatever is left in the buffer with the
        # last=True flag.
        payload = {"text": buffer, "last": True}
        await ws.send(json.dumps(payload))

    async def _passthrough_sender(
        self,
        text_iter: AsyncIterator[str],
        ws: Connection,
    ):
        """
        Simple mode: sends chunks immediately as they arrive. Every chunk is
        synthesized on its own, so for Silero each item of the iterator should
        be a meaningful phrase (or a complete SSML document when ssml=True),
        not a single LLM token.
        """
        async for chunk in text_iter:
            if chunk:
                await ws.send(json.dumps({"text": chunk}))

        # Signal end of stream.
        await ws.send(json.dumps({"last": True}))