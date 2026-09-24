from __future__ import annotations

import os
import asyncio
import inspect
import json
import mimetypes
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlencode
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    TypeVar,
    overload,
)

from websockets.exceptions import ConnectionClosed, InvalidStatus
from pydantic import BaseModel

from slet_sdk.aelite.tools import ToolBelt
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas import ErrorResponse, ErrorCode
from slet_sdk.schemas.errors import CallbackError, NetworkError, ProtocolError
from slet_sdk.aelite.typing import StreamMode
from slet_sdk.aelite.utils.device_info import get_device_string

if TYPE_CHECKING:
    from slet_sdk.aelite.manifest import AgentManifest, ToolConfig, ModelConfig
    from slet_sdk.aelite.resources.resource import AeliteResource


# ---------------------------------------------------------------------------
# Stream items returned to the user from stream()/on_incoming_stream
# ---------------------------------------------------------------------------


@dataclass
class TextDelta:
    """A chunk of the model's regular text response."""

    text: str


@dataclass
class ThinkingDelta:
    """A chunk of the model's thinking/reasoning text."""

    text: str


@dataclass
class StructuredResponse:
    """The final structured response from the agent (complies with the response_schema)."""

    data: dict[str, Any]


StreamItem = TextDelta | ThinkingDelta | StructuredResponse

_M = TypeVar("_M", bound=BaseModel)


# ---------------------------------------------------------------------------
# Internal queue markers
# ---------------------------------------------------------------------------


@dataclass
class _End:
    """Signal that a data stream in the queue has finished."""

    pass


@dataclass
class _Error:
    """Signal of an error in the queue."""

    error: ErrorResponse


@dataclass
class _MCPStdioProcess:
    """
    A locally running MCP server (stdio transport) plus the bookkeeping
    needed to correlate outgoing JSON-RPC requests with their responses.

    The server never sends us ``command``/``args`` - only ``server_id`` -
    so the mapping ``server_id → local process`` must be maintained here,
    on the client, exactly as documented in ``mcp_service.py``.
    """

    process: asyncio.subprocess.Process
    pending: dict[str, asyncio.Future] = field(default_factory=dict)
    reader_task: asyncio.Task | None = None
    next_id: int = 0


# Type of a queue item: either a stream chunk or a service marker.
_QueueItem = StreamItem | _End | _Error


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class AgentSession:
    """
    A session for communicating with an agent over WebSocket.

    Lifecycle
    ---------
    1. Create an instance via ``client.aelite.deploy_and_connect()``.
    2. Call ``await session.connect()`` - the connection is established,
       and the background listener starts automatically.
    3. Use ``chat()`` / ``stream()`` for dialogue.
    4. Register callbacks (``on_message``, ``on_error``, etc.)
       to receive unsolicited messages and notifications.
    5. Finish the work via ``await session.disconnect()``.

    Notes
    -----
    - "Request - Response" mode: ``chat()`` and ``stream()``. Each call gets
      its own ``msg`` identifier and is processed independently - you can
      call ``stream()``/``chat()`` concurrently, they don't block each other.
    - The response is split into regular text (``TextDelta``) and the
      model's thinking (``ThinkingDelta``), if the server/provider provides it.
      If a ``response_schema`` is passed to ``chat()``/``stream()``/``trigger()``,
      the agent's final answer is delivered as a single ``StructuredResponse``
      instead of free text.
    - Agent-initiated messages (broadcasts from triggers, including those
      launched by other devices of the same user) are routed via
      ``on_message`` (buffered, ``stream_unsolicited=False``) or
      ``on_incoming_stream`` (streamed, ``stream_unsolicited=True``), with
      one callback invocation per independent ``msg`` - concurrent
      unsolicited streams are not mixed with each other. Only one of these
      two callbacks is used at a time, depending on ``stream_unsolicited``.
      A structured response to a trigger is delivered via
      ``on_structured_response`` when ``stream_unsolicited=False``, and as
      an item of the ``on_incoming_stream`` generator when it is ``True``.
    - Automatic handling of client tool calls.
    - Automatic reconnect when the connection is dropped.
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

        # Child sub-agents, accessible via session.SubAgentId
        self.sub: dict[str, AgentSession] = {}

        # URLs
        ws_base = self._resource.base_ws_url
        base_url = self._resource.base_url
        self.device = device
        self.stream_mode = stream_mode
        self.http_url = base_url
        self.ws_url = ws_base
        params = {"stream_mode": StreamMode(stream_mode).value}
        if device:
            params["device"] = device
        self._ws_connect_url = (
            f"{self.ws_url}/agents/ws/{self.thread_id}?{urlencode(params)}"
        )

        self.headers = headers
        self.websocket = None
        self.logger = self._resource.logger
        self._is_connected = False

        self.logger.debug(f"Device name: {self.device}")
        self.logger.debug(f"Stream mode: {self.stream_mode}")

        # --- Behavior settings ---

        # If True: unsolicited messages are delivered via on_incoming_stream
        # as an async iterator (chunk by chunk, including StructuredResponse),
        # separately for each msg_id.
        # If False: text and thinking are accumulated in a buffer (also
        # separately per msg_id) and delivered whole via on_message; a
        # StructuredResponse is delivered via on_structured_response.
        self.stream_unsolicited: bool = False

        # --- Callbacks ---
        self._add_callbacks_attributes()

        # --- Client tool registry ---
        self._registered_tools: dict[str, Callable[..., Any]] = {}

        # --- MCP stdio server registry ---
        # server_id -> locally running process + pending JSON-RPC futures.
        # Populated via register_mcp_server(); consulted in _handle_mcp_call().
        self._mcp_stdio_servers: dict[str, _MCPStdioProcess] = {}

        # --- Internal machinery ---

        self._listener_task: asyncio.Task | None = None

        # Response queues for active chat/stream requests, one per msg_id.
        self._response_queues: dict[str, asyncio.Queue[_QueueItem]] = {}

        # Streaming queues for unsolicited messages, one per msg_id
        # (created upon receiving the first token for a new msg_id).
        self._unsolicited_streams: dict[str, asyncio.Queue[_QueueItem]] = {}

        # Buffers for unsolicited messages (stream_unsolicited=False),
        # one per msg_id.
        self._unsolicited_buffers: dict[str, list[StreamItem]] = {}

        # --- Reconnect settings ---
        self.max_reconnect_attempts: int = (
            10  # float("inf") for infinite reconnect
        )
        self.reconnect_delay: int = 5  # seconds between attempts
        self._reconnect_attempt: int = 0  # current attempt number (for logs only)
        self._manual_disconnect: bool = (
            False  # True = closed intentionally, no reconnect needed
        )

        # Event is cleared during reconnect; chat/stream/trigger wait on it.
        self._reconnect_event: asyncio.Event = asyncio.Event()
        self._reconnect_event.set()

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establishes a WebSocket connection and starts the background listener.

        Raises
        ------
        SletClientError
            If the server returned an error HTTP status during the handshake.
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
        Stops the background listener and closes the WebSocket connection.

        No reconnect is performed after this call.
        """
        self._manual_disconnect = True
        self._is_connected = False

        await self._cancel_listener()
        self._cleanup_queues(NetworkError(message="Disconnected by client."))

        for server_id in list(self._mcp_stdio_servers):
            await self.unregister_mcp_server(server_id)

        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from agent")

    async def disconnect_all(self) -> None:
        """Recursively disconnects the entire sub-agent tree, then the current session."""
        for sub_session in self.sub.values():
            await sub_session.disconnect_all()
        await self.disconnect()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tools(self, tools: list[Callable[..., Any] | ToolBelt]) -> None:
        """
        Registers client tool implementations that fulfil this agent's
        manifest contract.

        The manifest (``self.manifest.tools``) declares which client tools
        the LLM may call and their expected parameter schema - it is the
        contract. This method registers the local Python implementation that
        fulfils that contract; the implementation does not need to share any
        object with whatever declared the contract, it only needs to match
        it by name and parameters.

        If ``self.manifest`` is available, each registered tool is checked
        against the declared contract:
          - a client tool declared in the manifest but never registered here
            will fail at call time ("not registered on client");
          - a tool registered here but absent from the manifest will simply
            never be called by the LLM;
          - a name that matches but whose parameters don't (missing/extra
            required params, type mismatch) is logged as a conformance
            warning, since it will likely fail or misbehave on the first
            real call.

        Parameters
        ----------
        tools:
            A list of ``@tool``-decorated callables (sync or async) and/or
            ``ToolBelt`` instances. Each ``ToolBelt`` is flattened into its
            member tools automatically.
        """
        manifest_tools: list[ToolConfig] = self.manifest.tools if self.manifest is not None else []
        declared = {
            t.name: t for t in manifest_tools if t.type == "client"
        }

        for item in tools:
            if isinstance(item, ToolBelt):
                for fn in item:
                    self._register_single_tool(fn, declared)
            else:
                self._register_single_tool(item, declared)

        if declared:
            missing = declared.keys() - self._registered_tools.keys()
            for name in missing:
                self.logger.warning(
                    f"Client tool '{name}' is declared in the manifest but has no "
                    f"local implementation registered - calls to it will fail "
                    f"with 'not registered on client'."
                )

    def _register_single_tool(
            self,
            fn: Callable[..., Any],
            declared: dict[str, ToolConfig],
    ) -> None:
        name: str = getattr(fn, "_tool_name", None) or fn.__name__
        if name in self._registered_tools:
            self.logger.warning(
                f"Client tool '{name}' re-registered - overwriting previous handler."
            )

        contract = declared.get(name)
        if contract is not None:
            self._check_conformance(name, fn, contract)
        elif declared:  # manifest known, but this tool isn't in it
            self.logger.warning(
                f"Client tool '{name}' registered locally but not declared as a "
                f"client tool in the deployed manifest - the LLM will never call it."
            )

        self._registered_tools[name] = fn
        self.logger.debug(f"Registered client tool: {name}")

    def _check_conformance(
            self,
            name: str,
            fn: Callable[..., Any],
            contract: ToolConfig,
    ) -> None:
        """Compare the local implementation's parameters against the
        manifest's declared contract for this tool. Warns (does not raise)
        on mismatch, since the manifest is the source of truth and drift
        here is a bug to surface, not a hard failure to block on."""
        local_params: dict[str, dict[str, Any]] = getattr(fn, "_tool_params", None)
        if local_params is None:
            # fn wasn't decorated with @tool - we can't introspect its
            # contract-relevant metadata, so skip conformance checking.
            return

        contract_params = contract.params  # dict[str, ToolParam]

        missing_in_impl = contract_params.keys() - local_params.keys()
        extra_in_impl = local_params.keys() - contract_params.keys()
        if missing_in_impl:
            self.logger.warning(
                f"Client tool '{name}': manifest declares parameter(s) "
                f"{sorted(missing_in_impl)} that the local implementation does "
                f"not accept - calls using them will fail."
            )
        if extra_in_impl:
            self.logger.warning(
                f"Client tool '{name}': local implementation has parameter(s) "
                f"{sorted(extra_in_impl)} not declared in the manifest - the LLM "
                f"will never supply them, so make sure they have defaults."
            )

        for pname in contract_params.keys() & local_params.keys():
            c_type = contract_params[pname].type
            l_type = local_params[pname]["type"]
            if c_type != l_type:
                self.logger.warning(
                    f"Client tool '{name}', parameter '{pname}': manifest declares "
                    f"type '{c_type}' but local implementation expects '{l_type}'."
                )
            c_required = contract_params[pname].required
            l_required = local_params[pname]["required"]
            if c_required and not l_required:
                self.logger.warning(
                    f"Client tool '{name}', parameter '{pname}': manifest marks it "
                    f"required, but the local implementation has a default - this "
                    f"is harmless but suggests the contract and implementation "
                    f"were written independently and may drift further."
                )

    async def register_mcp_server(
            self,
            server_id: str,
            command: str,
            args: list[str] | None = None,
            env: dict[str, str] | None = None,
    ) -> None:
        """
        Launches a local MCP server as a stdio subprocess and registers it
        under ``server_id``.

        This is the client-side counterpart of ``MCPTransportStdio`` in the
        agent's manifest. The server only ever transmits ``server_id`` in
        ``mcp_call`` events (never ``command``, to avoid arbitrary remote
        command execution) - it is the client's job to know which local
        process that id refers to, which is exactly what this method sets up.

        Parameters
        ----------
        server_id:
            Must match ``MCPServerConfig.id`` in the deployed manifest.
        command:
            Executable to launch (e.g. ``"npx"``, ``"python"``).
        args:
            Arguments passed to ``command``.
        env:
            Extra environment variables merged into the subprocess's
            environment.

        Raises
        ------
        SletClientError
            If a server with this ``server_id`` is already registered.
        """
        if server_id in self._mcp_stdio_servers:
            raise SletClientError(
                CallbackError(
                    message=f"MCP server '{server_id}' is already registered. "
                            f"Call unregister_mcp_server() first."
                )
            )

        merged_env = {**os.environ, **(env or {})}
        process = await asyncio.create_subprocess_exec(
            command,
            *(args or []),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )

        entry = _MCPStdioProcess(process=process)
        self._mcp_stdio_servers[server_id] = entry
        entry.reader_task = asyncio.create_task(self._mcp_stdout_reader(server_id))

        self.logger.info(
            f"MCP stdio server '{server_id}' started: {command} {' '.join(args or [])}"
        )

    async def unregister_mcp_server(self, server_id: str) -> None:
        """
        Terminates and removes a previously registered MCP stdio server.

        Safe to call for an unknown ``server_id`` - it's a no-op in that case.
        """
        entry = self._mcp_stdio_servers.pop(server_id, None)
        if entry is None:
            return

        if entry.reader_task and not entry.reader_task.done():
            entry.reader_task.cancel()

        # Unblock any in-flight mcp_call waiting on this process.
        for fut in entry.pending.values():
            if not fut.done():
                fut.set_exception(
                    SletClientError(NetworkError(message=f"MCP server '{server_id}' unregistered."))
                )

        if entry.process.returncode is None:
            entry.process.terminate()
            try:
                await asyncio.wait_for(entry.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                entry.process.kill()

        self.logger.info(f"MCP stdio server '{server_id}' stopped.")

    # ------------------------------------------------------------------
    # Low-level sending
    # ------------------------------------------------------------------

    async def send(self, message: str) -> None:
        """
        Sends a raw message over the WebSocket.

        If a reconnect is currently in progress - waits for it to finish.

        Parameters
        ----------
        message:
            The string to send (text or JSON).

        Raises
        ------
        SletClientError
            If the connection is not established.
        """
        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(NetworkError(message="WebSocket is not connected."))

        await self.websocket.send(message)

    # ------------------------------------------------------------------
    # Public request methods
    # ------------------------------------------------------------------

    async def stream(
        self,
        message: str,
        files: list[Any] | None = None,
        response_schema: dict | type[BaseModel] | None = None,
    ) -> AsyncGenerator[StreamItem, None]:
        """
        Sends a message to the agent and returns an async generator of
        response items (``TextDelta`` / ``ThinkingDelta`` / ``StructuredResponse``).

        Each call gets its own ``msg`` and is processed independently of
        other active ``stream()``/``chat()`` calls - they can be called
        concurrently.

        Parameters
        ----------
        message:
            The user's message text.
        files:
            A list of files to attach. Each file can be ``str``,
            ``bytes``, or a file-like object.
        response_schema:
            JSON Schema (dict, type='object') or a pydantic class. If specified,
            the agent completes the response by calling final-tool, and the stream receives
            one element of ``StructuredResponse`` (free text is not streamed).

        Yields
        ------
        StreamItem
            ``TextDelta`` - a chunk of the regular response text.
            ``ThinkingDelta`` - a chunk of the model's thinking text (if any).
            ``StructuredResponse`` - the complete structured response
            (arrives once, at the end).

        Raises
        ------
        SletClientError
            On a connection error or if the agent/server returned an error
            (including a msg conflict - practically impossible with UUID4,
            but would still correctly surface as an exception).
        """
        schema_dict = (
            self._normalize_schema(response_schema)
            if response_schema is not None
            else None
        )

        attachments: list[dict] = []
        if files:
            for f in files:
                res = await self.upload_file(f)
                attachments.append({"ref": f"upload:{res['file_id']}"})

        await self._reconnect_event.wait()

        if not self._is_connected:
            raise SletClientError(
                NetworkError(message="Cannot send message: not connected.")
            )

        msg_id = str(uuid.uuid4())
        queue: asyncio.Queue[_QueueItem] = asyncio.Queue()
        self._response_queues[msg_id] = queue

        try:
            payload: dict[str, Any] = {
                "type": "message",
                "msg": msg_id,
                "text": message,
            }
            if attachments:
                payload["attachments"] = attachments
            if schema_dict is not None:
                payload["response_schema"] = schema_dict
            await self.send(json.dumps(payload))

            while True:
                item = await queue.get()
                match item:
                    case TextDelta() | ThinkingDelta() | StructuredResponse():
                        yield item
                    case _End():
                        break
                    case _Error(error):
                        raise SletClientError(error)
        finally:
            self._response_queues.pop(msg_id, None)

    @overload
    async def chat(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: None = None,
    ) -> str: ...

    @overload
    async def chat(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: type[_M],
    ) -> _M: ...

    @overload
    async def chat(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: dict,
    ) -> dict: ...

    @overload
    async def chat(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: dict | type[BaseModel],
    ) -> dict | BaseModel: ...

    async def chat(
        self,
        message: str,
        files: list[Any] | None = None,
        response_schema: dict | type[BaseModel] | None = None,
    ) -> str | dict | BaseModel:
        """
        Sends a message and returns the agent's full response.

        Without ``response_schema`` returns the accumulated text (no thinking).
        With ``response_schema`` returns the structured result: a ``dict``, or
        a model instance if a pydantic class was passed.

        Parameters
        ----------
        message:
            The user's message text.
        files:
            A list of files to attach.
        response_schema:
            JSON Schema (dict, type='object') or a pydantic class.

        Returns
        -------
        str | dict | BaseModel
            The agent's full response.
        """
        chunks: list[str] = []
        structured: dict | None = None
        async for item in self.stream(
            message=message, files=files, response_schema=response_schema
        ):
            if isinstance(item, TextDelta):
                chunks.append(item.text)
            elif isinstance(item, StructuredResponse):
                structured = item.data

        return self._finalize_structured(structured, response_schema)

    @overload
    async def chat_with_thinking(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: None = None,
    ) -> tuple[str, str]: ...

    @overload
    async def chat_with_thinking(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: type[_M],
    ) -> tuple[_M, str]: ...

    @overload
    async def chat_with_thinking(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: dict,
    ) -> tuple[dict, str]: ...

    @overload
    async def chat_with_thinking(
        self, message: str, files: list[Any] | None = None, *,
        response_schema: dict | type[BaseModel],
    ) -> tuple[dict | BaseModel, str]: ...

    async def chat_with_thinking(
        self,
        message: str,
        files: list[Any] | None = None,
        response_schema: dict | type[BaseModel] | None = None,
    ) -> tuple[str | dict | BaseModel, str]:
        """
        Like ``chat()``, but additionally returns the accumulated
        model thinking text.

        Parameters
        ----------
        message:
            The user's message text.
        files:
            A list of files to attach.
        response_schema:
            JSON Schema (dict, type='object') or a pydantic class. If given,
            the first element of the result is the structured response
            (a ``dict``, or a model instance if a pydantic class was passed)
            instead of the accumulated text.

        Returns
        -------
        tuple[str | dict | BaseModel, str]
            (response, thinking text). The second element is an empty string
            if the provider/agent did not supply thinking.
        """
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        structured: dict | None = None
        async for item in self.stream(
            message=message, files=files, response_schema=response_schema
        ):
            if isinstance(item, TextDelta):
                text_chunks.append(item.text)
            elif isinstance(item, ThinkingDelta):
                thinking_chunks.append(item.text)
            elif isinstance(item, StructuredResponse):
                structured = item.data

        thinking = "".join(thinking_chunks)

        if response_schema is None:
            return "".join(text_chunks), thinking

        return self._finalize_structured(structured, response_schema), thinking

    async def upload_file(self, file: Any) -> dict:
        """
        Uploads a file to the server over HTTP.

        Parameters
        ----------
        file:
            - ``str``       - text content (text/plain, UTF-8).
            - ``bytes``     - raw bytes (application/octet-stream).
            - file-like     - an object with a ``.read()`` method
              (``BufferedReader``, ``SpooledTemporaryFile``, etc.).

        Returns
        -------
        dict
            The server response, containing at least ``file_id``.

        Raises
        ------
        TypeError
            If the argument type is not supported.
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

    async def trigger(
        self,
        name: str,
        payload: dict[str, Any] | None = None,
        response_schema: dict | type[BaseModel] | None = None,
    ) -> str:
        """
        Sends a trigger (event) to the agent without waiting for a response.

        The response to the trigger (and its errors) is broadcast to all
        devices of the session, including the initiator - use the
        ``on_message`` / ``on_structured_response`` (buffered mode) or
        ``on_incoming_stream`` (streamed mode), and ``on_server_error``
        callbacks to receive it. The returned ``msg_id`` can be used to
        match it with these callbacks.

        Parameters
        ----------
        name:
            The trigger name.
        payload:
            Arbitrary event data.
        response_schema:
            JSON Schema (dict, type='object') or a pydantic class.

        Returns
        -------
        str
            The generated message identifier (``msg``) - for later
            matching with unsolicited events.

        Raises
        ------
        ConnectionError
            If the WebSocket is not connected.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected.")

        msg_id = str(uuid.uuid4())
        event: dict[str, Any] = {
            "type": "trigger",
            "msg": msg_id,
            "name": name,
            "payload": payload or {},
        }
        if response_schema is not None:
            event["response_schema"] = self._normalize_schema(response_schema)
        await self.send(json.dumps(event))
        return msg_id

    async def set_stream_mode(self, stream_mode: StreamMode) -> None:
        """
        Changes the streaming mode of messages from the agent.

        Does not affect requests already in progress,
        takes effect on the next request.

        Parameters
        ----------
        stream_mode:
            The message streaming mode.

        Raises
        ------
        ConnectionError
            If the WebSocket is not connected.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected.")

        event = {"type": "set_stream_mode", "mode": stream_mode}
        await self.send(json.dumps(event))

    async def set_models(self, models: list[ModelConfig]) -> None:
        """
        Supplies this connection's own models fallback chain, entirely replacing
        the agent's manifest-level ``models`` chain for requests sent on this
        connection - the two are never merged.

        Only accepted by agents deployed with
        ``AgentManifest.model_rotation.allow_client_override=True``. If the
        agent's manifest also leaves ``models`` empty, calling this is
        mandatory before the first ``chat()``/``stream()``/``trigger()`` call,
        since the server has no default chain to fall back to.

        Does not affect requests already in progress; takes effect on the
        next request. Sent as a dedicated WebSocket event rather than a query
        parameter, since ``ModelConfig`` entries may carry an ``api_key``.

        Parameters
        ----------
        models:
            The models fallback chain to use for this connection, in priority
            order. Must not be empty.

        Raises
        ------
        ConnectionError
            If the WebSocket is not connected.
        """
        if not self._is_connected:
            raise ConnectionError("Not connected.")

        event = {
            "type": "set_models",
            "models": [m.model_dump(mode="json") for m in models],
        }
        await self.send(json.dumps(event))

    # ------------------------------------------------------------------
    # Helper private methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_schema(schema: dict | type[BaseModel]) -> dict:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_json_schema()
        if isinstance(schema, dict):
            return schema
        raise TypeError(
            f"response_schema must be a JSON Schema dict or a pydantic model class, "
            f"got {type(schema).__name__}."
        )

    @staticmethod
    def _finalize_structured(
        structured: dict | None,
        response_schema: dict | type[BaseModel],
    ) -> dict | BaseModel:
        if structured is None:
            raise SletClientError(
                ProtocolError(
                    message=(
                        "Agent finished without a structured response. "
                        "The server may not support response_schema."
                    )
                )
            )
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            return response_schema.model_validate(structured)
        return structured

    def _report_protocol_error(self, message: str) -> None:
        """
        Reports a protocol violation (malformed data from the server) via
        ``on_error`` without interrupting the read loop.
        """
        self.logger.warning(message)
        self._invoke_callback(self.on_error, SletClientError(ProtocolError(message=message)))

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
        Adds attributes for callbacks.
        Callbacks support both async and sync functions.
        """

        # Called when a full unsolicited message is received
        # (stream_unsolicited=False only). Signature: (text, thinking, msg_id).
        # thinking - an empty string if there was no thinking.
        # Not invoked when stream_unsolicited=True (use on_incoming_stream
        # instead). Also not invoked if the response contained neither text
        # nor thinking (e.g. only a StructuredResponse; see
        # on_structured_response).
        self.on_message: (
            Callable[[str, str, str], None | Awaitable[None]]
        ) | None = None

        # Called when a structured response to an unsolicited request (trigger)
        # is received (stream_unsolicited=False only). Signature: (data, msg_id).
        # Invoked at the end of the answer, after on_message (if any).
        # Not invoked when stream_unsolicited=True: StructuredResponse is then
        # delivered as an item of the on_incoming_stream generator.
        self.on_structured_response: (
            Callable[[dict, str | None], None | Awaitable[None]]
        ) | None = None

        # Called when a new unsolicited stream starts
        # (stream_unsolicited=True only). Signature: (msg_id, generator).
        # The generator yields every StreamItem (TextDelta, ThinkingDelta,
        # StructuredResponse). Not invoked when stream_unsolicited=False
        # (use on_message instead).
        self.on_incoming_stream: (
            Callable[[str, AsyncIterable[StreamItem]], None | Awaitable[None]]
        ) | None = None

        # Notification that a tool has started on the server side.
        # Signature: (tool_name, msg_id).
        self.on_tool_start: (
            Callable[[str, str], None | Awaitable[None]]
        ) | None = None

        # Global exception handler.
        self.on_error: (Callable[[Exception], None | Awaitable[None]]) | None = None

        # Global handler for error messages from the server.
        # Signature: (error, msg_id). msg_id may be None if the error
        # is not tied to a specific message.
        self.on_server_error: (
            Callable[[ErrorResponse, str | None], None | Awaitable[None]]
        ) | None = None

        # Notification of a models/provider change.
        self.on_model_change: (
            Callable[[str, str, str], None | Awaitable[None]]
        ) | None = None

    async def _cancel_listener(self) -> None:
        """
        Cancels the background listener and waits for it to finish.

        Safe to call even if the listener has already finished or never started.
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
        Guarantees the old listener is stopped and a new one is started.

        Called after a successful reconnect to prevent a situation where
        two listeners simultaneously read from the same socket.
        """
        await self._cancel_listener()
        self._listener_task = asyncio.create_task(self._listen_loop())

    def _cleanup_queues(self, error: NetworkError | None = None) -> None:
        """
        Unblocks everyone waiting on connection loss and clears the buffers.

        If ``error`` is given, active ``stream()``/``chat()`` calls (and
        unsolicited streams) are terminated with that error instead of a
        normal end-of-stream, so a dropped connection is never mistaken for
        a complete (or truncated but "successful") response.
        """
        def marker() -> _QueueItem:
            return _Error(error) if error is not None else _End()

        for queue in self._response_queues.values():
            queue.put_nowait(marker())

        for queue in self._unsolicited_streams.values():
            queue.put_nowait(marker())
        self._unsolicited_streams.clear()
        self._unsolicited_buffers.clear()

    # ------------------------------------------------------------------
    # Background WebSocket read loop
    # ------------------------------------------------------------------

    async def _listen_loop(self) -> None:
        """
        Continuously reads the WebSocket and routes incoming data.

        All protocol messages are JSON with a ``type`` field. Routing is
        done by event type and by the ``msg`` field (if present): if
        ``msg`` matches an active ``stream()``/``chat()`` - it goes to its
        queue, otherwise it's handled as unsolicited (broadcast from a
        trigger, including one launched by another device).

        On connection loss, starts ``_reconnect_loop``,
        unless ``disconnect()`` was called manually.
        """
        self.logger.debug("Background listener started.")
        try:
            async for raw_msg in self.websocket:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError as e:
                    self._report_protocol_error(
                        f"Received a message that is not valid JSON: {e}. "
                        f"Raw (truncated): {raw_msg[:200]!r}"
                    )
                    continue

                if not isinstance(data, dict) or "type" not in data:
                    self._report_protocol_error(
                        f"Received a message of unknown format: {str(data)[:200]!r}"
                    )
                    continue

                try:
                    await self._dispatch_event(data)
                except Exception as e:
                    err = ProtocolError(
                        message=f"Failed to process event {data.get('type')!r}: {e}"
                    )
                    self.logger.warning(err.message)
                    self._invoke_callback(self.on_error, SletClientError(err))
                    queue = self._response_queues.get(data.get("msg"))
                    if queue is not None:
                        queue.put_nowait(_Error(err))

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
        Handles a single protocol event (JSON with a ``type`` field).

        Parameters
        ----------
        data:
            The parsed JSON event object.
        """
        event_type: str = data.get("type", "")
        msg_id: str | None = data.get("msg")

        if event_type == "chunk":
            await self._route_item(msg_id, TextDelta(data.get("chunk", "")))

        elif event_type == "thinking":
            await self._route_item(msg_id, ThinkingDelta(data.get("chunk", "")))

        elif event_type == "response":
            await self._route_item(msg_id, TextDelta(data.get("payload", "")))

        elif event_type == "thinking_response":
            await self._route_item(msg_id, ThinkingDelta(data.get("payload", "")))

        elif event_type == "structured_response":
            await self._route_item(msg_id, StructuredResponse(data=data.get("payload", {})))

        elif event_type == "end_of_answer":
            await self._route_end(msg_id)

        elif event_type == "error":
            try:
                err = ErrorResponse(**data.get("payload", {}))
            except Exception as e:
                err = ProtocolError(message=f"Malformed error payload from server: {e}")
            await self._route_error(msg_id, err)

        elif event_type == "client_tool_call":
            asyncio.create_task(self._handle_client_tool_call(data.get("payload", {})))

        elif event_type == "mcp_call":
            asyncio.create_task(self._handle_mcp_call(data.get("payload", {})))

        elif event_type == "tool_start":
            name: str = data.get("payload", {}).get("name", "")
            self._invoke_callback(self.on_tool_start, name, msg_id)

        elif event_type == "model_changed":
            payload = data.get("payload", {})
            self._invoke_callback(
                self.on_model_change,
                payload.get("provider"),
                payload.get("models"),
                payload.get("source", "server"),
            )

        elif event_type == "stream_mode_changed":
            pass  # confirmation of mode change, no dedicated callback yet

        else:
            self.logger.debug(f"Unknown event type: {event_type!r}")

    async def _route_item(self, msg_id: str | None, item: StreamItem) -> None:
        """
        Routes a response chunk
        (text/thinking/structured response) either to the queue of an
        active stream()/chat(), or to unsolicited handling, depending on
        whether this ``msg_id`` is registered locally.
        """
        queue = self._response_queues.get(msg_id) if msg_id else None
        if queue is not None:
            await queue.put(item)
            return

        await self._handle_unsolicited_item(msg_id, item)

    async def _route_end(self, msg_id: str | None) -> None:
        """Routes the end-of-response signal to the appropriate queue/buffer."""
        queue = self._response_queues.get(msg_id) if msg_id else None
        if queue is not None:
            await queue.put(_End())
            return

        if msg_id is None:
            return

        if self.stream_unsolicited:
            stream_queue = self._unsolicited_streams.pop(msg_id, None)
            if stream_queue is not None:
                await stream_queue.put(_End())
        else:
            items = self._unsolicited_buffers.pop(msg_id, [])
            if items:
                text = "".join(i.text for i in items if isinstance(i, TextDelta))
                thinking = "".join(i.text for i in items if isinstance(i, ThinkingDelta))
                if text or thinking:
                    self._invoke_callback(self.on_message, text, thinking, msg_id)
                for i in items:
                    if isinstance(i, StructuredResponse):
                        self._invoke_callback(self.on_structured_response, i.data, msg_id)

    async def _route_error(self, msg_id: str | None, error: ErrorResponse) -> None:
        """
        Routes an error to the appropriate queue/stream (to wake up waiters)
        and always additionally invokes the global ``on_server_error``.
        """
        queue = self._response_queues.get(msg_id) if msg_id else None
        if queue is not None:
            await queue.put(_Error(error))
        elif msg_id is not None:
            stream_queue = self._unsolicited_streams.pop(msg_id, None)
            if stream_queue is not None:
                await stream_queue.put(_Error(error))
            self._unsolicited_buffers.pop(msg_id, None)

        self._invoke_callback(self.on_server_error, error, msg_id)

    async def _handle_unsolicited_item(self, msg_id: str | None, item: StreamItem) -> None:
        """
        Routes a chunk of an unsolicited message into a stream or buffer,
        separately for each ``msg_id``.

        Parameters
        ----------
        msg_id:
            The message identifier from the server. If the server for some
            reason did not send a ``msg`` - the event is logged and
            dropped, since it cannot be correctly grouped with the rest.
        item:
            A chunk of text/thinking or the structured response from the agent.
        """
        if msg_id is None:
            self._report_protocol_error("Unsolicited event without msg - dropped.")
            return

        if self.stream_unsolicited:
            queue = self._unsolicited_streams.get(msg_id)
            if queue is None:
                queue = asyncio.Queue()
                self._unsolicited_streams[msg_id] = queue
                gen = self._make_unsolicited_generator(queue)
                self._invoke_callback(self.on_incoming_stream, msg_id, gen)

            await queue.put(item)
        else:
            self._unsolicited_buffers.setdefault(msg_id, []).append(item)

    async def _on_listener_failure(self, error: NetworkError) -> None:
        """
        Called when ``_listen_loop`` terminates unexpectedly.

        Resets state, invokes ``on_error``, and starts reconnecting,
        unless the connection was closed manually.

        Parameters
        ----------
        error:
            Description of the failure cause.
        """
        self._is_connected = False
        self._cleanup_queues(error)

        if not self._manual_disconnect:
            if self.on_error:
                self._invoke_callback(self.on_error, SletClientError(error))
            # Start the reconnect as a separate task.
            # Use add_done_callback so the exception is not silently lost.
            task = asyncio.create_task(self._reconnect_loop())
            task.add_done_callback(self._on_reconnect_task_done)

    def _on_reconnect_task_done(self, task: asyncio.Task) -> None:
        """
        Callback for the completion of the reconnect task.

        Retrieves the exception from the task so it doesn't get
        silently swallowed.
        """
        if not task.cancelled():
            exc = task.exception()
            if exc:
                self.logger.error(f"Reconnect loop failed with exception: {exc}")

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _serialize_tool_result(self, value: Any) -> str:
        """Convert an arbitrary client-tool return value into text for the
        server. Strings pass through untouched; everything else is JSON-encoded
        where possible (dicts/lists/most dataclasses via ``default=str``), and
        falls back to ``str()`` only if that's not possible at all."""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)

    def _invoke_callback(self, callback: Callable[..., Any] | None, *args: Any) -> None:
        """
        Safely invokes a callback in the current event loop.

        - Async functions are scheduled via ``asyncio.create_task()``.
        - Sync functions are called directly.

        .. warning::
            Synchronous callbacks run in the event loop thread.
            Blocking operations in them will stall ``_listen_loop``.
            Use async callbacks whenever possible.

        Parameters
        ----------
        callback:
            The callable or ``None``.
        *args:
            Arguments for the callback.
        """
        if not callback:
            return

        try:
            if inspect.iscoroutinefunction(callback):
                asyncio.create_task(callback(*args))
            else:
                name = getattr(callback, "__name__", repr(callback))
                self.logger.debug(
                    f"{name} is a sync callback. "
                    "Make sure it does not perform any blocking operations."
                )
                callback(*args)

        except Exception as e:
            self.logger.error(f"Error invoking callback {callback}: {e}")

            # Avoid recursion: don't call on_error from on_error itself
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
    ) -> AsyncGenerator[StreamItem, None]:
        """
        A generator that reads chunks of an unsolicited message from the queue.

        Terminates upon receiving ``_End`` or raises ``SletClientError``
        on ``_Error``.

        Parameters
        ----------
        queue:
            The queue into which ``_dispatch_event`` puts items belonging
            to a specific ``msg_id``.

        Yields
        ------
        StreamItem
            ``TextDelta`` / ``ThinkingDelta`` / ``StructuredResponse``.
        """
        while True:
            item = await queue.get()
            match item:
                case TextDelta() | ThinkingDelta() | StructuredResponse():
                    yield item
                case _End():
                    break
                case _Error(error):
                    raise SletClientError(error)

    async def _handle_client_tool_call(self, payload: dict) -> None:
        """
        Handles a client tool call initiated by the agent.

        Looks up the registered tool by name, calls it,
        and sends the result back to the server.

        Parameters
        ----------
        payload:
            A dict with the fields ``name``, ``args``, and ``tool_call_id``.
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
                    raw_result = await fn(**args)
                else:
                    loop = asyncio.get_running_loop()
                    # noinspection PyTypeChecker
                    raw_result = await loop.run_in_executor(None, lambda: fn(**args))
                result = self._serialize_tool_result(raw_result)
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

    async def _mcp_stdout_reader(self, server_id: str) -> None:
        """
        Continuously reads newline-delimited JSON-RPC responses from an MCP
        stdio process's stdout and resolves the matching pending future by
        the response's ``id``.

        Terminates when the process closes stdout (crash or normal exit);
        any requests still awaiting a response at that point are failed
        instead of hanging forever.
        """
        entry = self._mcp_stdio_servers.get(server_id)
        if entry is None:
            return

        try:
            while True:
                line = await entry.process.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    self.logger.warning(
                        f"MCP server '{server_id}' sent a non-JSON line: {line!r}"
                    )
                    continue

                resp_id = str(data.get("id"))
                fut = entry.pending.pop(resp_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(data)

        except Exception as e:
            self.logger.error(f"MCP stdio reader for '{server_id}' crashed: {e}")

        finally:
            for fut in entry.pending.values():
                if not fut.done():
                    fut.set_exception(
                        SletClientError(
                            NetworkError(message=f"MCP server '{server_id}' stdout closed.")
                        )
                    )

    async def _handle_mcp_call(self, payload: dict) -> None:
        """
        Handles an ``mcp_call`` event initiated by the agent.

        Forwards a JSON-RPC 2.0 request to the local stdio process
        registered under ``payload['server_id']``, waits for the matching
        response, and sends the result back to the server as ``mcp_result``.

        Parameters
        ----------
        payload:
            Dict with ``server_id``, ``method``, ``params``, ``call_id``,
            and ``pod_id`` (echoed back verbatim so the server routes the
            response to the pod that is actually waiting on it).
        """
        server_id: str = payload.get("server_id", "")
        method: str = payload.get("method", "")
        params: dict = payload.get("params", {})
        call_id: str = payload.get("call_id", "")
        pod_id = payload.get("pod_id")

        entry = self._mcp_stdio_servers.get(server_id)
        result: dict

        if entry is None:
            result = {"error": {"message": f"MCP server '{server_id}' not registered on client."}}
            self.logger.warning(result["error"]["message"])
        else:
            entry.next_id += 1
            rpc_id = str(entry.next_id)
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            entry.pending[rpc_id] = fut

            request = {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}

            try:
                entry.process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
                await entry.process.stdin.drain()
                response = await asyncio.wait_for(fut, timeout=30)
                # Сервер (mcp_service.py) читает именно "result" объект
                # напрямую (result.get("tools", [])/["contents"]/["content"]),
                # поэтому пробрасываем содержимое result "как есть".
                result = response.get("result") or response.get("error", {})
            except asyncio.TimeoutError:
                result = {"error": {"message": f"MCP call '{method}' on '{server_id}' timed out."}}
                self.logger.warning(result["error"]["message"])
            except Exception as e:
                result = {"error": {"message": str(e)}}
                self.logger.error(f"MCP call to '{server_id}' failed: {e}")
            finally:
                # noinspection PyAsyncCall
                entry.pending.pop(rpc_id, None)

        response_event = {
            "type": "mcp_result",
            "payload": {
                "call_id": call_id,
                "result": result,
                "pod_id": pod_id,
            },
        }

        if self._is_connected:
            await self.websocket.send(json.dumps(response_event))

    # ------------------------------------------------------------------
    # Reconnect
    # ------------------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        """
        Attempts to restore the WebSocket connection after a disconnect.

        Performs up to ``max_reconnect_attempts`` attempts with a pause
        of ``reconnect_delay`` seconds between them.

        Invokes ``on_error`` on each failure.
        Restarts ``_listen_loop`` after a successful reconnect.

        Raises
        ------
        SletClientError
            If all attempts are exhausted.
        """
        self._reconnect_attempt = 0
        self._reconnect_event.clear()  # block chat/stream/trigger during reconnect

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

                # Guarantee the old listener is stopped (in case it's still alive)
                # and start a new one. Without this, two listeners could
                # simultaneously read the same socket and interleave messages.
                await self._restart_listener()

                self._reconnect_event.set()  # unblock chat/stream/trigger
                self.logger.info("WebSocket successfully reconnected.")
                return  # ✅ Success

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
                    # Unblock waiters so they don't hang forever
                    self._reconnect_event.set()
                    raise SletClientError(error)

                await asyncio.sleep(self.reconnect_delay)

    # ------------------------------------------------------------------
    # Magic: attribute-based access to sub-agents
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> AgentSession:
        """
        Magic attribute-based access to child sub-agents.

        Example usage::

            session.SubAgentId.connect()
            await session.SubAgentId.chat("hello")

        Parameters
        ----------
        name:
            The sub-agent identifier.

        Raises
        ------
        AttributeError
            If no sub-agent with this name is registered.
        """
        # Pass through private and dunder attributes normally to avoid recursion
        if name.startswith("_"):
            raise AttributeError(name)

        if name in self.sub:
            return self.sub[name]

        raise AttributeError(
            f"Sub-agent '{name}' not found. "
            f"Available: {list(self.sub.keys()) or 'none registered'}."
        )