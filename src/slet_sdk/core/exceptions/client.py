from enum import Enum
from slet_sdk.core.schemas import ErrorResponse


class SletClientError(Exception):
    """
    Исключение для использования Slet Client
    Slet Client это клиент для работы с API Slet сервера
    """

    def __init__(self, error: ErrorResponse):
        self.status = error.status
        self.error = error.error
        self.message = error.message
        self.extra = error.extra
        self.trace_id = error.trace_id
        super().__init__(error.message)

    def to_dict(self, stringify_error: bool = False) -> dict:
        if (
            stringify_error
        ):  # Чтобы добавить extra, trace_id только если он есть, и преобразовать error в строку (т.к. как он может быть enum)
            if isinstance(self.error, Enum):
                error = self.error.value
            else:
                error = str(self.error)

            result = {
                "status": self.status,
                "error": error,
                "message": self.message,
            }

            if self.extra:
                result["extra"] = self.extra
            if self.trace_id:
                result["trace_id"] = self.trace_id

            return result

        return {
            "status": self.status,
            "error": self.error,
            "message": self.message,
            "extra": self.extra,
            "trace_id": self.trace_id,
        }

    def __str__(self):
        return f"{self.__class__.__name__} {self.to_dict(stringify_error=True)}"
