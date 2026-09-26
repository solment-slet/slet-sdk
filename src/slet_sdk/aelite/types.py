from typing import Union
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TTSStreamConfig:
    """
    Audio format metadata sent by the Piper TTS server as the very first
    message on the WebSocket session (the "config" event).

    Attributes:
        sample_rate (int): Samples per second, e.g. 22050.
        sample_width (int): Bytes per sample (2 for int16 PCM).
        channels (int): Number of audio channels (1 = mono).
    """

    sample_rate: int
    sample_width: int
    channels: int

# What `stream_tts` can yield: either the format descriptor (once, first)
# or raw PCM audio chunks (bytes, many times).
TTSStreamItem = Union[TTSStreamConfig, bytes]


class StreamMode(str, Enum):
    """
    Streaming modes are sent when a WebSocket connection is established or within a connection.
    """

    tokens = "tokens"  # by tokens
    message = "message"  # the full LLM text in one piece
    silent = "silent"  # only the events of thul and end_of_answer, without text
