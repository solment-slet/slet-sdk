"""
test_mcp_stdio_session.py - tests for AgentSession's stdio MCP mechanism.

Why stdio, and why this is different from the "e2e requires an open port"
problem discussed for SSE MCP servers:

    A real MCP server reachable over SSE needs its own publicly reachable
    endpoint that the backend (aelite-service) can dial into - that's
    infrastructure this test environment can't stand up for CI.

    stdio is the opposite shape: MCPServerConfig.transport=MCPTransportStdio
    only ever tells the *backend* a server_id (never `command`, per its
    docstring - "The server identifies the MCP server exclusively by id").
    The backend sends an `mcp_call` event over the *existing* WebSocket
    connection, and it's the already-connected client (this SDK, via
    AgentSession.register_mcp_server / _handle_mcp_call) that owns the
    actual subprocess and speaks JSON-RPC to it over stdin/stdout. No new
    inbound network path is ever opened - the local subprocess is only
    ever reachable from the same process that spawned it. That's exactly
    what makes stdio testable here: we launch a trivial local MCP-like
    script as a subprocess, register it under a server_id via the SDK, and
    exercise the whole path either fully offline (talking to
    AgentSession's stdio machinery directly, no WS at all) or e2e (through
    a real backend + WS connection, which only needs the WS endpoint to be
    reachable - already true for every other e2e test in this suite).

Two groups:

1. Offline unit tests against AgentSession's stdio machinery directly
   (register_mcp_server, unregister_mcp_server, _handle_mcp_call,
   _mcp_stdout_reader) - a real local subprocess is spawned (that's not
   "the network", it's just a CLI process), but no WebSocket/backend is
   involved at all. Mirrors TestHandleClientToolCall in test_tools.py.

2. End-to-end tests that deploy a real manifest with an MCPTransportStdio
   server via client/models fixtures, connect over a real WS backend, and
   drive tool use so the backend actually emits `mcp_call` and consumes
   `mcp_result` for real.
"""

from __future__ import annotations

import asyncio
import json
import logging

import sys
import textwrap
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from slet_sdk.aelite.manifest import (
    AgentManifest,
    MCPServerConfig,
    MCPTransportStdio,
    MemoryConfig,
)
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.exceptions import SletClientError


# ===========================================================================
# Fixture: a trivial local "MCP-like" stdio script
# ===========================================================================
#
# This is NOT a spec-complete MCP server - it only implements enough of the
# JSON-RPC shape that AgentSession._handle_mcp_call / _mcp_stdout_reader
# actually depend on (newline-delimited JSON-RPC 2.0 over stdin/stdout,
# correlated by `id`), so the SDK's plumbing can be exercised without
# pulling in a real MCP server implementation as a test dependency.
#
# Methods implemented:
#   initialize   -> {}
#   tools/list   -> {"tools": [{"name": "echo", ...}]}
#   tools/call   -> echoes back params.arguments.text, or raises for
#                   method == "slow" to test the client-side timeout path
#   crash        -> exits immediately without responding, to test the
#                   "stdout closed while a call is pending" path

_STDIO_SCRIPT = textwrap.dedent(
    """
    import sys, json, os

    EVIDENCE_PATH = os.environ.get("MCP_TEST_EVIDENCE_PATH")

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    def record_call(method, params):
        # Proof that THIS local process actually received the call -
        # independent of anything the LLM might claim in its response
        # text. Appended (not overwritten), so tests can also assert
        # call ordering/count if needed.
        if not EVIDENCE_PATH:
            return
        with open(EVIDENCE_PATH, "a") as f:
            f.write(json.dumps({"method": method, "params": params}) + "\\n")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue

        rpc_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": rpc_id, "result": {}})
        elif method == "tools/list":
            send({
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echoes text",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string",
                                        "description": "Text to echo back",
                                    }
                                },
                                "required": ["text"],
                            },
                        }
                    ]
                },
            })
        elif method == "tools/call" and params.get("name") == "echo":
            record_call(method, params)
            text = (params.get("arguments") or {}).get("text", "")
            send({
                "jsonrpc": "2.0", "id": rpc_id,
                "result": {"content": [{"type": "text", "text": text}]},
            })
        elif method == "tools/call" and params.get("name") == "boom":
            record_call(method, params)
            send({
                "jsonrpc": "2.0", "id": rpc_id,
                "error": {"message": "tool 'boom' always fails"},
            })
        elif method == "slow":
            # Deliberately never responds, to exercise the client's
            # asyncio.wait_for(timeout=30) path in a bounded test by
            # monkeypatching that timeout down (see test below).
            continue
        elif method == "crash":
            sys.exit(1)
        else:
            send({"jsonrpc": "2.0", "id": rpc_id, "error": {"message": f"unknown method {method}"}})
    """
).strip()


@pytest.fixture
def stdio_script(tmp_path: Path) -> Path:
    script_path = tmp_path / "fake_mcp_server.py"
    script_path.write_text(_STDIO_SCRIPT)
    return script_path


@pytest.fixture
def mcp_call_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Path where the fake stdio script records every tools/call it actually
    received. This is the ONLY reliable way to confirm the MCP tool was
    genuinely invoked end-to-end - asserting on the LLM's final chat text
    is not sufficient, since a models can produce a plausible-sounding
    "I echoed X" response without the tool call ever reaching the process
    (e.g. if the tool wasn't in its schema, the call was rejected, or it
    simply hallucinated success).
    """
    evidence_path = tmp_path / "mcp_calls.jsonl"
    monkeypatch.setenv("MCP_TEST_EVIDENCE_PATH", str(evidence_path))
    return evidence_path


def _read_evidence(evidence_path: Path) -> list[dict]:
    if not evidence_path.exists():
        return []
    return [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]


# ===========================================================================
# Helpers (mirrors test_tools.py's offline session setup)
# ===========================================================================


class _FakeResource:
    base_ws_url = "ws://fake-host"
    base_url = "http://fake-host"

    def __init__(self):
        self.logger = logging.getLogger("test.aelite.mcp_stdio")
        self.websockets = MagicMock()


def _make_disconnected_session(manifest: AgentManifest | None = None) -> AgentSession:
    return AgentSession(
        thread_id="unit-test-thread",
        headers={},
        manifest=manifest,
        resource=_FakeResource(),
    )


async def _connected_fake_session() -> AgentSession:
    """A session with _is_connected=True and a mocked websocket, so
    _handle_mcp_call's final `await self.websocket.send(...)` doesn't
    explode - mirrors TestHandleClientToolCall._run_call in test_tools.py."""
    session = _make_disconnected_session(None)
    session._is_connected = True
    session.websocket = AsyncMock()
    return session


# ===========================================================================
# 1. register_mcp_server / unregister_mcp_server
# ===========================================================================


class TestRegisterMcpServer:
    async def test_register_starts_a_real_subprocess(self, stdio_script):
        session = _make_disconnected_session(None)
        try:
            await session.register_mcp_server(
                server_id="local",
                command=sys.executable,
                args=[str(stdio_script)],
            )
            assert "local" in session._mcp_stdio_servers
            entry = session._mcp_stdio_servers["local"]
            assert entry.process.returncode is None  # still running
            assert entry.reader_task is not None and not entry.reader_task.done()
        finally:
            await session.unregister_mcp_server("local")

    async def test_duplicate_server_id_raises(self, stdio_script):
        session = _make_disconnected_session(None)
        try:
            await session.register_mcp_server(
                server_id="local", command=sys.executable, args=[str(stdio_script)]
            )
            with pytest.raises(SletClientError, match="already registered"):
                await session.register_mcp_server(
                    server_id="local", command=sys.executable, args=[str(stdio_script)]
                )
        finally:
            await session.unregister_mcp_server("local")

    async def test_unregister_unknown_server_id_is_a_safe_noop(self):
        session = _make_disconnected_session(None)
        await session.unregister_mcp_server("never-registered")  # must not raise

    async def test_unregister_terminates_the_process(self, stdio_script):
        session = _make_disconnected_session(None)
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        process = session._mcp_stdio_servers["local"].process

        await session.unregister_mcp_server("local")

        assert "local" not in session._mcp_stdio_servers
        # terminate() was requested; give the loop a beat to reap it
        await asyncio.wait_for(process.wait(), timeout=5)
        assert process.returncode is not None

    async def test_unregister_cancels_the_reader_task(self, stdio_script):
        session = _make_disconnected_session(None)
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        reader_task = session._mcp_stdio_servers["local"].reader_task

        await session.unregister_mcp_server("local")

        assert reader_task.cancelled() or reader_task.done()

    async def test_disconnect_tears_down_all_registered_mcp_servers(self, stdio_script):
        session = _make_disconnected_session(None)
        session._is_connected = True
        session.websocket = AsyncMock()

        await session.register_mcp_server(
            server_id="a", command=sys.executable, args=[str(stdio_script)]
        )
        await session.register_mcp_server(
            server_id="b", command=sys.executable, args=[str(stdio_script)]
        )

        await session.disconnect()

        assert session._mcp_stdio_servers == {}


# ===========================================================================
# 2. _handle_mcp_call - the JSON-RPC round trip over a real local process
# ===========================================================================


class TestHandleMcpCall:
    async def _run_call(
            self, session: AgentSession, server_id: str, method: str, params: dict, call_id: str = "call-1"
    ) -> dict:
        """Runs one mcp_call and returns its result payload. Safe to call
        concurrently on the same session - captures the specific send() call
        matching this call_id instead of relying on the mock's global
        call count, which races when multiple _run_call invocations share
        the same session.websocket mock."""
        sends_before = session.websocket.send.await_count
        await session._handle_mcp_call(
            {
                "server_id": server_id,
                "method": method,
                "params": params,
                "call_id": call_id,
                "pod_id": "pod-A",
            }
        )
        # Find the specific send() that carries this call_id, rather than
        # asserting a delta of exactly 1 - concurrent calls interleave freely.
        matching = [
            json.loads(c.args[0])
            for c in session.websocket.send.await_args_list[sends_before:]
            if json.loads(c.args[0]).get("payload", {}).get("call_id") == call_id
        ]
        assert len(matching) == 1
        sent = matching[0]
        assert sent["type"] == "mcp_result"
        assert sent["payload"]["pod_id"] == "pod-A"
        return sent["payload"]["result"]

    async def test_tools_list_round_trip(self, stdio_script):
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            result = await self._run_call(session, "local", "tools/list", {})
            assert result == {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echoes text",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "Text to echo back",
                                }
                            },
                            "required": ["text"],
                        },
                    }
                ]
            }
        finally:
            await session.unregister_mcp_server("local")

    async def test_tools_call_echo_round_trip(self, stdio_script):
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            result = await self._run_call(
                session, "local", "tools/call",
                {"name": "echo", "arguments": {"text": "hello mcp"}},
            )
            assert result == {"content": [{"type": "text", "text": "hello mcp"}]}
        finally:
            await session.unregister_mcp_server("local")

    async def test_multiple_calls_are_correlated_by_id_independently(self, stdio_script):
        """Two concurrent calls to the same process must each get their
        own response, not cross-resolve - this is what entry.pending[id]
        keyed by rpc_id (not call order) is protecting against."""
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            results = await asyncio.gather(
                self._run_call(
                    session, "local", "tools/call",
                    {"name": "echo", "arguments": {"text": "first"}}, call_id="c1",
                ),
                self._run_call(
                    session, "local", "tools/call",
                    {"name": "echo", "arguments": {"text": "second"}}, call_id="c2",
                ),
            )
            texts = {r["content"][0]["text"] for r in results}
            assert texts == {"first", "second"}
        finally:
            await session.unregister_mcp_server("local")

    async def test_tool_error_from_server_is_forwarded_as_result_error(self, stdio_script):
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            result = await self._run_call(
                session, "local", "tools/call", {"name": "boom", "arguments": {}}
            )
            # _handle_mcp_call does `result = response.get("result") or
            # response.get("error", {})` - on an error response it assigns the
            # error object itself into `result`, it does not re-wrap it under
            # an extra "error" key.
            assert result == {"message": "tool 'boom' always fails"}
        finally:
            await session.unregister_mcp_server("local")

    async def test_unregistered_server_id_returns_error_without_crashing(self):
        session = await _connected_fake_session()
        result = await self._run_call(session, "never-registered", "tools/list", {})
        assert "not registered on client" in result["error"]["message"]

    async def test_call_times_out_when_process_never_responds(self, stdio_script, monkeypatch):
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            # The real timeout is a hardcoded 30s inside _handle_mcp_call;
            # patch asyncio.wait_for to fail fast so this test doesn't
            # actually block for 30 seconds.
            real_wait_for = asyncio.wait_for

            async def fast_wait_for(fut, timeout):
                return await real_wait_for(fut, timeout=0.3)

            monkeypatch.setattr(asyncio, "wait_for", fast_wait_for)

            result = await self._run_call(session, "local", "slow", {})
            assert "timed out" in result["error"]["message"]
        finally:
            await session.unregister_mcp_server("local")

    async def test_process_crash_fails_pending_calls_instead_of_hanging(self, stdio_script):
        session = await _connected_fake_session()
        await session.register_mcp_server(
            server_id="local", command=sys.executable, args=[str(stdio_script)]
        )
        try:
            result = await self._run_call(session, "local", "crash", {})
            # _handle_mcp_call catches the exception from the failed future
            # (set by _mcp_stdout_reader's finally-block once stdout closes)
            # and reports it as a result error rather than propagating.
            assert "error" in result
        finally:
            await session.unregister_mcp_server("local")


# ===========================================================================
# 3. End-to-end - real backend, real WS, local stdio MCP server
# ===========================================================================
#
# No open inbound port anywhere: the backend only ever knows `server_id`
# (see MCPServerConfig docstring), and dispatches `mcp_call` over the
# already-open WebSocket connection this session establishes outbound to
# the backend. The subprocess itself is never network-reachable.


async def test_e2e_agent_uses_stdio_mcp_tool(client, models, stdio_script, mcp_call_evidence):
    # Unique per-test-run marker - makes the assertion specific to THIS
    # call reaching the process, not just "some echo call happened at
    # some point" (relevant since evidence files could theoretically be
    # shared/stale across a flaky rerun).
    marker_text = f"ping-{uuid.uuid4().hex[:8]}"

    manifest = AgentManifest(
        id="TestAgentStdioMCP",
        system_prompt=(
            "You have access to tools provided by an MCP server named "
            "'local'. If the user asks you to echo some text, call the "
            "'local_echo' tool with that text immediately."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        mcp_servers=[
            MCPServerConfig(
                id="local",
                transport=MCPTransportStdio(command=sys.executable, args=[str(stdio_script)]),
                load_tools=True,
            )
        ],
    )

    session = await client.aelite.get_session_from_manifest(manifest)

    tool_started: list[str] = []
    session.on_tool_start = lambda name, msg_id: tool_started.append(name)

    await session.register_mcp_server(
        server_id="local", command=sys.executable, args=[str(stdio_script)]
    )
    try:
        result = await session.chat(f"Please echo the text '{marker_text}'.")
        print("\n" + result)
        assert isinstance(result, str)
        assert result

        # --- Proof #1: the local subprocess actually received tools/call.
        # This is independent of anything the models says in `result` - if
        # the tool call never reached the process, this file stays empty
        # regardless of how confident the models's text response sounds.
        calls = _read_evidence(mcp_call_evidence)
        echo_calls = [
            c for c in calls
            if c["method"] == "tools/call" and c["params"].get("name") == "echo"
        ]
        assert echo_calls, (
            f"No 'echo' tools/call reached the local MCP process. "
            f"All recorded calls: {calls}"
        )
        called_texts = [
            (c["params"].get("arguments") or {}).get("text", "") for c in echo_calls
        ]
        assert any(marker_text in t for t in called_texts), (
            f"Local process received echo call(s), but none carried our "
            f"marker text {marker_text!r}: {called_texts}"
        )

        # --- Proof #2: the server-side graph actually reached a tool node
        # for this tool (protocol-level signal, independent of both the
        # models's text and the subprocess's own bookkeeping).
        assert any("echo" in name for name in tool_started), (
            f"on_tool_start never fired for the echo tool. Fired for: {tool_started}"
        )
    finally:
        await session.unregister_mcp_server("local")


async def test_e2e_stdio_server_survives_disconnect_cleanup(client, models, stdio_script):
    """After session.disconnect(), the locally spawned MCP process must
    actually be torn down (not leaked) - exercised end-to-end through the
    full connect/deploy/disconnect lifecycle rather than calling
    unregister_mcp_server directly."""
    manifest = AgentManifest(
        id="TestAgentStdioMCPCleanup",
        system_prompt="You have access to MCP tools from server 'local'.",
        models=models,
        memory=MemoryConfig(enabled=False),
        mcp_servers=[
            MCPServerConfig(
                id="local",
                transport=MCPTransportStdio(command=sys.executable, args=[str(stdio_script)]),
            )
        ],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    await session.register_mcp_server(
        server_id="local", command=sys.executable, args=[str(stdio_script)]
    )
    process = session._mcp_stdio_servers["local"].process

    await session.disconnect()

    await asyncio.wait_for(process.wait(), timeout=5)
    assert process.returncode is not None
    assert session._mcp_stdio_servers == {}