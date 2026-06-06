from enum import Enum


class StreamMode(str, Enum):
    """
    Режимы стриминга, отправляются при установке WebSocket соединения или внутри соединения.
    """

    tokens = "tokens"  # по токенам
    message = "message"  # полный текст LLM одним куском (on_chat_model_end)
    silent = "silent"  # только события тулов и end_of_answer, без текста
