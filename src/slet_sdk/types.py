from collections.abc import Mapping, Iterable
from typing import (
    Protocol,
    Any,
    AsyncContextManager,
    Literal,
)


class LoggerLike(Protocol):
    def debug(self, msg: str, *args, **kwargs) -> None: ...
    def info(self, msg: str, *args, **kwargs) -> None: ...
    def warning(self, msg: str, *args, **kwargs) -> None: ...
    def error(self, msg: str, *args, **kwargs) -> None: ...
    def exception(self, msg: str, *args, **kwargs) -> None: ...
    def critical(self, msg: str, *args, **kwargs) -> None: ...


class WebSocketClient(Protocol):
    async def send(self, data: Any) -> None: ...
    async def recv(self) -> Any: ...


class WebsocketsModule(Protocol):
    async def connect(
        self,
        uri: str,
        *,
        ping_interval: float | None = None,
        ping_timeout: float | None = None,
        close_timeout: float | None = None,
        additional_headers: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
        **kwargs: Any,
    ) -> AsyncContextManager[WebSocketClient]: ...


UserPrivilege = Literal["user", "dev", "admin"]
