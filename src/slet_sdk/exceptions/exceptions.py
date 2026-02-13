# TODO Удалить этот файл, теперь используется только SletClientError

import json
from slet_sdk.schemas import ErrorCode
from .client import SletClientError


class NetworkError(SletClientError):
    """
    Проблемы сети (таймаут, DNS, соединение)
    """

    def __init__(
        self,
        status: int = 599,
        error: str = ErrorCode.NETWORK_ERROR,
        message: str = "Network Error",
    ):
        super().__init__(status=status, error=error, message=message)


class ApiError(Exception):
    """
    Базовое исключение для всех HTTP-запросов
    """

    def __init__(
        self,
        status: int,
        error: str = ErrorCode.UNKNOWN_ERROR,
        message: str | None = None,
        extra: dict | None = None,
        raw_json: (
            dict | str | None
        ) = None,  # Можно передать необработанный json, он автоматически преобразуется
    ):
        super().__init__(error)

        # Если у нас есть необработанный json то извлекаем данные из него
        if raw_json:
            if isinstance(raw_json, str):  # JSON строки превращаем в словарь
                raw_json = json.loads(raw_json)

            self.status = raw_json.get("status", status)
            self.error = raw_json.get("error", error)
            self.message = raw_json.get("message", message or "")
            self.extra = raw_json.get("extra", extra or {})

        else:  # Или используем готовое
            self.status = status
            self.error = error
            self.message = message or ""
            self.extra = extra or {}

    def to_dict(self) -> dict:
        result = {
            "status": self.status,
            "error": self.error,
            "message": self.message,
        }
        if self.extra:  # проверка на True
            result["extra"] = self.extra
        return result

    def __str__(self):
        return f"{self.__class__.__name__} {self.to_dict()}"


class UnauthorizedError(ApiError):
    """Ошибка авторизации (401)"""

    def __init__(
        self,
        status: int = 401,
        error: str = ErrorCode.UNAUTHORIZED,
        message: str = "Unauthorized",
        raw_json: dict | str | None = None,
    ):
        super().__init__(status=status, error=error, message=message, raw_json=raw_json)


class InternalServerError(ApiError):
    """Ошибка сервера (5xx)"""

    def __init__(
        self,
        status: int = 500,
        error: str = ErrorCode.INTERNAL_SERVER_ERROR,
        message: str = "Internal Server Error",
        raw_json: dict | str | None = None,
    ):
        super().__init__(status=status, error=error, message=message, raw_json=raw_json)


class UnexpectedStatusError(ApiError):
    """Любой другой неожиданный HTTP-код"""

    def __init__(self, status_code: int, message: str = None):
        self.status_code = status_code
        self.message = message or f"Unexpected status code {status_code}"
        super().__init__(self.message)
