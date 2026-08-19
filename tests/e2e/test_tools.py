"""
test_tools.py - tests for client-tool declaration (``tools.py``), their
integration into ``AgentManifest`` (``manifest.py``), and their execution
inside ``AgentSession`` (``session.py``).

Split into three groups:

1. Pure unit tests for ``@tool`` / ``ToolBelt`` (no network, no fixtures).
2. Manifest-level validation tests (duplicate names, timeouts, broadcast
   config, flattening of ``ToolBelt``/bare tools/dicts/``ToolConfig``).
3. ``AgentSession`` tests - some against a fake in-memory resource
   (no real WebSocket, exercising ``register_tools`` conformance checks and
   ``_handle_client_tool_call`` directly), and some end-to-end against a
   live agent via the ``client``/``models`` fixtures (mirrors the existing
   ``test_client_tools`` example).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock

import pytest

from slet_sdk.aelite.manifest import (
    AgentManifest,
    BroadcastConfig,
    BroadcastMode,
    ConcurrencyConfig,
    MemoryConfig,
    ToolConfig,
    ToolParam, SequentialExecutionConfig,
)
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.aelite.tools import ToolBelt, tool


# ===========================================================================
# Helpers
# ===========================================================================


class _FakeResource:
    """Minimal stand-in for ``AeliteResource`` sufficient to construct an
    ``AgentSession`` without opening a real WebSocket connection."""

    base_ws_url = "ws://fake-host"
    base_url = "http://fake-host"

    def __init__(self):
        self.logger = logging.getLogger("test.aelite.fake_resource")
        self.websockets = MagicMock()


def _make_disconnected_session(manifest: AgentManifest | None = None) -> AgentSession:
    """Builds an ``AgentSession`` that was never actually connected, for
    unit-testing registration/dispatch logic in isolation."""
    return AgentSession(
        thread_id="unit-test-thread",
        headers={},
        manifest=manifest,
        resource=_FakeResource(),
    )


# ===========================================================================
# 1. @tool decorator - unit tests
# ===========================================================================


class TestBareToolDecorator:
    def test_bare_usage_uses_docstring(self):
        @tool
        def get_clipboard() -> str:
            """Returns the user's current clipboard contents."""
            return "clip"

        assert get_clipboard._tool_name == "get_clipboard"
        assert get_clipboard._tool_description == "Returns the user's current clipboard contents."
        assert get_clipboard._tool_params == {}
        assert get_clipboard._is_client_tool is True

    def test_parens_no_args_behaves_like_bare(self):
        @tool()
        def ping() -> str:
            """Pings something."""
            return "pong"

        assert ping._tool_name == "ping"
        assert ping._tool_description == "Pings something."

    def test_explicit_name_and_description_override_docstring(self):
        @tool(name="custom_name", description="Custom description.")
        def some_fn() -> str:
            """This docstring should be ignored."""
            return "x"

        assert some_fn._tool_name == "custom_name"
        assert some_fn._tool_description == "Custom description."

    def test_description_is_stripped(self):
        @tool(description="   padded description   ")
        def fn() -> str:
            return "x"

        assert fn._tool_description == "padded description"

    def test_missing_description_and_docstring_raises(self):
        with pytest.raises(ValueError, match="has no description"):
            @tool
            def undocumented(x: str) -> str:
                return x

    def test_unsupported_extra_kwarg_raises_typeerror(self):
        with pytest.raises(TypeError):
            @tool(description="desc", not_a_real_field=True)
            def fn() -> str:
                return "x"

    def test_allowed_extra_kwargs_timeout_and_broadcast(self):
        @tool(description="desc", timeout=45, broadcast=BroadcastConfig(mode=BroadcastMode.all))
        def fn() -> str:
            return "x"

        assert fn._tool_extra == {"timeout": 45, "broadcast": BroadcastConfig(mode=BroadcastMode.all)}

    def test_unsupported_param_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="unsupported"):
            @tool(description="desc")
            def fn(x: dict) -> str:
                return str(x)

    def test_self_and_cls_are_skipped(self):
        class Foo:
            @tool(description="method as tool")
            def bar(self, x: str) -> str:
                return x

        assert "self" not in Foo.bar._tool_params
        assert "x" in Foo.bar._tool_params

        class FooCls:
            @tool(description="classmethod-ish tool")
            def bar(cls, x: int) -> str:
                return str(x)

        assert "cls" not in FooCls.bar._tool_params


class TestToolParameters:
    def test_no_args_tool(self):
        @tool(description="No args")
        def get_time() -> str:
            return datetime.now(timezone.utc).isoformat()

        assert get_time._tool_params == {}

    def test_single_required_arg(self):
        @tool(description="Echoes text")
        def echo(text: str) -> str:
            return text

        params = echo._tool_params
        assert params["text"] == {
            "type": "str",
            "description": None,
            "required": True,
            "default": None,
        }

    def test_single_optional_arg_with_default(self):
        @tool(description="Greets someone")
        def greet(name: str = "world") -> str:
            return f"hello {name}"

        params = greet._tool_params
        assert params["name"]["required"] is False
        assert params["name"]["default"] == "world"

    def test_multiple_mixed_required_and_optional_args(self):
        @tool(description="Books a flight")
        def book_flight(
            destination: str,
            passengers: int,
            refundable: bool = False,
            max_price: float = 500.0,
        ) -> str:
            return f"{destination}:{passengers}:{refundable}:{max_price}"

        params = book_flight._tool_params
        assert set(params.keys()) == {"destination", "passengers", "refundable", "max_price"}

        assert params["destination"]["type"] == "str"
        assert params["destination"]["required"] is True

        assert params["passengers"]["type"] == "int"
        assert params["passengers"]["required"] is True

        assert params["refundable"]["type"] == "bool"
        assert params["refundable"]["required"] is False
        assert params["refundable"]["default"] is False

        assert params["max_price"]["type"] == "float"
        assert params["max_price"]["required"] is False
        assert params["max_price"]["default"] == 500.0

    def test_annotated_param_carries_description(self):
        @tool(description="Sets volume")
        def set_volume(
            level: Annotated[int, "Volume level from 0 to 100"],
        ) -> None:
            ...

        assert set_volume._tool_params["level"]["description"] == "Volume level from 0 to 100"
        assert set_volume._tool_params["level"]["type"] == "int"

    def test_annotated_param_without_string_metadata_has_none_description(self):
        @tool(description="Weird annotation")
        def fn(x: Annotated[str, object()]) -> str:
            return x

        assert fn._tool_params["x"]["description"] is None

    def test_untyped_param_defaults_to_str(self):
        @tool(description="Untyped param")
        def fn(x) -> str:
            return str(x)

        assert fn._tool_params["x"]["type"] == "str"

    def test_all_four_supported_types(self):
        @tool(description="All types")
        def fn(a: str, b: int, c: float, d: bool) -> str:
            return f"{a}{b}{c}{d}"

        params = fn._tool_params
        assert params["a"]["type"] == "str"
        assert params["b"]["type"] == "int"
        assert params["c"]["type"] == "float"
        assert params["d"]["type"] == "bool"


# ===========================================================================
# 2. ToolBelt - unit tests
# ===========================================================================


class TestToolBelt:
    def test_register_and_iterate(self):
        belt = ToolBelt()

        @belt.tool
        def get_clipboard() -> str:
            """Returns clipboard contents."""
            return "clip"

        @belt.tool(description="Shows a notification")
        def show_notification(title: str, message: str) -> None:
            ...

        assert len(belt) == 2
        names = {fn._tool_name for fn in belt}
        assert names == {"get_clipboard", "show_notification"}

    def test_duplicate_name_within_belt_raises(self):
        belt = ToolBelt(name="my_belt")

        @belt.tool(description="first")
        def do_thing() -> str:
            return "a"

        with pytest.raises(ValueError, match="already registered"):
            @belt.tool(name="do_thing", description="second, same name")
            def do_thing_2() -> str:
                return "b"

    def test_duplicate_name_error_mentions_belt_name(self):
        belt = ToolBelt(name="named_belt")

        @belt.tool(description="first")
        def dup() -> str:
            return "a"

        with pytest.raises(ValueError, match="named_belt"):
            @belt.tool(name="dup", description="second")
            def dup_again() -> str:
                return "b"

    def test_two_independent_belts_do_not_collide(self):
        belt_a = ToolBelt()
        belt_b = ToolBelt()

        @belt_a.tool(description="A's version")
        def shared_name() -> str:
            return "a"

        @belt_b.tool(description="B's version")
        def shared_name() -> str:  # noqa: F811
            return "b"

        assert len(belt_a) == 1
        assert len(belt_b) == 1
        assert next(iter(belt_a))._tool_description == "A's version"
        assert next(iter(belt_b))._tool_description == "B's version"

    def test_repr_lists_tool_names(self):
        belt = ToolBelt(name="repr_belt")

        @belt.tool(description="d")
        def alpha() -> str:
            return "a"

        r = repr(belt)
        assert "repr_belt" in r
        assert "alpha" in r

    def test_belt_tool_supports_multi_arg_signatures(self):
        belt = ToolBelt()

        @belt.tool(description="Books a hotel")
        def book_hotel(city: str, nights: int, breakfast: bool = True) -> str:
            return f"{city}:{nights}:{breakfast}"

        fn = next(iter(belt))
        assert set(fn._tool_params) == {"city", "nights", "breakfast"}
        assert fn._tool_params["breakfast"]["required"] is False


# ===========================================================================
# 3. Manifest-level tool integration - unit tests
# ===========================================================================


class TestManifestToolNormalization:
    def test_bare_tool_flattened_into_manifest(self):
        @tool(description="A bare tool")
        def bare_fn() -> str:
            return "x"

        manifest = AgentManifest(id="A", tools=[bare_fn])
        assert len(manifest.tools) == 1
        assert manifest.tools[0].name == "bare_fn"
        assert manifest.tools[0].type == "client"

    def test_toolbelt_flattened_into_manifest(self):
        belt = ToolBelt()

        @belt.tool(description="one")
        def t1() -> str:
            return "1"

        @belt.tool(description="two")
        def t2(x: int) -> str:
            return str(x)

        manifest = AgentManifest(id="A", tools=[belt])
        names = {t.name for t in manifest.tools}
        assert names == {"t1", "t2"}
        for t in manifest.tools:
            assert t.type == "client"

    def test_dict_and_toolconfig_entries_accepted(self):
        raw = {"name": "server_side", "type": "server", "description": "does stuff"}
        cfg = ToolConfig(name="another", type="server", description="also stuff")

        manifest = AgentManifest(id="A", tools=[raw, cfg])
        names = {t.name for t in manifest.tools}
        assert names == {"server_side", "another"}

    def test_mixed_sources_all_flattened_together(self):
        belt = ToolBelt()

        @belt.tool(description="belt tool")
        def belt_fn() -> str:
            return "x"

        @tool(description="bare tool")
        def bare_fn() -> str:
            return "y"

        manifest = AgentManifest(
            id="A",
            tools=[belt, bare_fn, {"name": "srv", "type": "server", "description": "d"}],
        )
        names = {t.name for t in manifest.tools}
        assert names == {"belt_fn", "bare_fn", "srv"}

    def test_duplicate_tool_name_across_sources_raises(self):
        @tool(description="one")
        def dup() -> str:
            return "a"

        belt = ToolBelt()

        @belt.tool(name="dup", description="two, same name, different belt")
        def dup2() -> str:
            return "b"

        with pytest.raises(ValueError, match="Duplicate tool name"):
            AgentManifest(id="A", tools=[dup, belt])

    def test_unsupported_tool_entry_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported entry"):
            AgentManifest(id="A", tools=[object()])

    def test_client_tool_timeout_must_be_less_than_processing_timeout(self):
        @tool(description="slow tool", timeout=120)
        def slow() -> str:
            return "x"

        with pytest.raises(ValueError, match="must be strictly less"):
            AgentManifest(
                id="A",
                concurrency=ConcurrencyConfig(
                    mode="sequential",
                    sequential=SequentialExecutionConfig(processing_timeout=120)
                ),
                tools=[slow],
            )

    def test_client_tool_timeout_ok_when_strictly_less(self):
        @tool(description="fast enough", timeout=30)
        def fast() -> str:
            return "x"

        manifest = AgentManifest(
            id="A",
            concurrency = ConcurrencyConfig(
                mode="sequential",
                sequential=SequentialExecutionConfig(processing_timeout=60)
            ),
            tools=[fast],
        )
        assert manifest.tools[0].timeout == 30


class TestBroadcastConfig:
    def test_default_broadcast_is_disabled(self):
        @tool(description="a tool")
        def fn() -> str:
            return "x"

        manifest = AgentManifest(id="A", tools=[fn])
        assert manifest.tools[0].broadcast.mode == BroadcastMode.all

    def test_broadcast_set_via_tool_decorator_extra_kwarg(self):
        @tool(
            description="Ask every connected client",
            broadcast=BroadcastConfig(mode=BroadcastMode.collect, collect_window=2.0, include_client_id=True),
        )
        def poll_all_clients() -> str:
            return "x"

        manifest = AgentManifest(id="A", tools=[poll_all_clients])
        cfg = manifest.tools[0].broadcast
        assert cfg.mode == BroadcastMode.collect
        assert cfg.collect_window == 2.0
        assert cfg.include_client_id is True

    def test_broadcast_threshold_percent_out_of_range_raises(self):
        with pytest.raises(ValueError, match="threshold_percent"):
            BroadcastConfig(mode=BroadcastMode.threshold, threshold_percent=0)

        with pytest.raises(ValueError, match="threshold_percent"):
            BroadcastConfig(mode=BroadcastMode.threshold, threshold_percent=150)

    def test_broadcast_threshold_percent_boundary_is_valid(self):
        cfg = BroadcastConfig(mode=BroadcastMode.threshold, threshold_percent=100)
        assert cfg.threshold_percent == 100

    def test_broadcast_first_mode_directly_on_toolconfig(self):
        cfg = ToolConfig(
            name="ping_anyone",
            type="client",
            description="Pings any single connected client",
            broadcast=BroadcastConfig(mode=BroadcastMode.first),
        )
        assert cfg.broadcast.mode == BroadcastMode.first


# ===========================================================================
# 4. AgentSession - offline unit tests (no real WebSocket)
# ===========================================================================


class TestRegisterToolsConformance:
    def test_register_tool_declared_and_implemented_no_warnings(self, caplog):
        @tool(description="matches contract")
        def get_time() -> str:
            return "now"

        manifest = AgentManifest(id="A", tools=[get_time])
        session = _make_disconnected_session(manifest)

        with caplog.at_level(logging.WARNING):
            session.register_tools([get_time])

        assert "get_time" in session._registered_tools
        assert not any("get_time" in r.message for r in caplog.records if r.levelno >= logging.WARNING)

    def test_declared_but_not_registered_warns(self, caplog):
        @tool(description="declared only")
        def only_declared() -> str:
            return "x"

        manifest = AgentManifest(id="A", tools=[only_declared])
        session = _make_disconnected_session(manifest)

        with caplog.at_level(logging.WARNING):
            session.register_tools([])

        assert any(
            "not registered on client" in r.message or "no local implementation" in r.message
            for r in caplog.records
        )

    def test_registered_but_not_declared_warns(self, caplog):
        other = ToolConfig(name="other_tool", type="client", description="d")
        manifest = AgentManifest(id="A", tools=[other])
        session = _make_disconnected_session(manifest)

        @tool(description="never declared in manifest")
        def rogue_tool() -> str:
            return "x"

        with caplog.at_level(logging.WARNING):
            session.register_tools([rogue_tool])

        assert any("never call it" in r.message for r in caplog.records)

    def test_param_type_mismatch_warns(self, caplog):
        declared = ToolConfig(
            name="set_level",
            type="client",
            description="set a level",
            params={"level": ToolParam(type="int", required=True)},
        )
        manifest = AgentManifest(id="A", tools=[declared])
        session = _make_disconnected_session(manifest)

        @tool(name="set_level", description="local impl with wrong type")
        def set_level(level: str) -> str:
            return level

        with caplog.at_level(logging.WARNING):
            session.register_tools([set_level])

        assert any("type" in r.message and "set_level" in r.message for r in caplog.records)

    def test_missing_param_in_impl_warns(self, caplog):
        declared = ToolConfig(
            name="book_flight",
            type="client",
            description="books a flight",
            params={
                "destination": ToolParam(type="str", required=True),
                "passengers": ToolParam(type="int", required=True),
            },
        )
        manifest = AgentManifest(id="A", tools=[declared])
        session = _make_disconnected_session(manifest)

        @tool(name="book_flight", description="only handles destination")
        def book_flight(destination: str) -> str:
            return destination

        with caplog.at_level(logging.WARNING):
            session.register_tools([book_flight])

        assert any("passengers" in r.message for r in caplog.records)

    def test_extra_param_in_impl_warns(self, caplog):
        declared = ToolConfig(
            name="greet",
            type="client",
            description="greets",
            params={"name": ToolParam(type="str", required=True)},
        )
        manifest = AgentManifest(id="A", tools=[declared])
        session = _make_disconnected_session(manifest)

        @tool(name="greet", description="local impl with an extra optional param")
        def greet(name: str, loudly: bool = False) -> str:
            return name

        with caplog.at_level(logging.WARNING):
            session.register_tools([greet])

        assert any("loudly" in r.message for r in caplog.records)

    def test_required_in_manifest_but_optional_locally_warns(self, caplog):
        declared = ToolConfig(
            name="fn",
            type="client",
            description="d",
            params={"x": ToolParam(type="str", required=True)},
        )
        manifest = AgentManifest(id="A", tools=[declared])
        session = _make_disconnected_session(manifest)

        @tool(name="fn", description="local has a default")
        def fn(x: str = "default") -> str:
            return x

        with caplog.at_level(logging.WARNING):
            session.register_tools([fn])

        assert any("required" in r.message and "default" in r.message for r in caplog.records)

    def test_toolbelt_registration_via_session(self):
        belt = ToolBelt()

        @belt.tool(description="one")
        def t1() -> str:
            return "1"

        @belt.tool(description="two")
        def t2(x: int) -> str:
            return str(x)

        manifest = AgentManifest(id="A", tools=[belt])
        session = _make_disconnected_session(manifest)
        session.register_tools([belt])

        assert set(session._registered_tools) == {"t1", "t2"}

    def test_re_registering_same_name_warns_and_overwrites(self, caplog):
        session = _make_disconnected_session(None)

        @tool(name="dup", description="first impl")
        def first_impl() -> str:
            return "first"

        @tool(name="dup", description="second impl")
        def second_impl() -> str:
            return "second"

        with caplog.at_level(logging.WARNING):
            session.register_tools([first_impl])
            session.register_tools([second_impl])

        assert session._registered_tools["dup"] is second_impl
        assert any("re-registered" in r.message for r in caplog.records)

    def test_no_manifest_skips_conformance_checks(self):
        session = _make_disconnected_session(None)

        @tool(description="no manifest to check against")
        def fn() -> str:
            return "x"

        # Should not raise even though there's no manifest to compare against.
        session.register_tools([fn])
        assert "fn" in session._registered_tools

    def test_plain_callable_without_metadata_registers_without_conformance_check(self):
        declared = ToolConfig(name="plain", type="client", description="d")
        manifest = AgentManifest(id="A", tools=[declared])
        session = _make_disconnected_session(manifest)

        def plain(x: str) -> str:
            return x

        # Not @tool-decorated -> no _tool_params -> conformance check skipped safely.
        session.register_tools([plain])
        assert session._registered_tools["plain"] is plain


class TestHandleClientToolCall:
    async def _run_call(self, session: AgentSession, name: str, args: dict, call_id: str = "call-1"):
        session._is_connected = True
        session.websocket = AsyncMock()
        await session._handle_client_tool_call(
            {"name": name, "args": args, "tool_call_id": call_id}
        )
        assert session.websocket.send.await_count == 1
        sent = session.websocket.send.await_args[0][0]
        import json

        return json.loads(sent)

    async def test_sync_tool_executes_and_returns_result(self):
        session = _make_disconnected_session(None)

        @tool(description="adds two numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        session.register_tools([add])

        response = await self._run_call(session, "add", {"a": 2, "b": 3})
        assert response["payload"]["result"] == "5"
        assert response["payload"]["tool_call_id"] == "call-1"

    async def test_async_tool_executes_and_returns_result(self):
        session = _make_disconnected_session(None)

        @tool(description="async greeting")
        async def async_greet(name: str) -> str:
            await asyncio.sleep(0)
            return f"hello {name}"

        session.register_tools([async_greet])

        response = await self._run_call(session, "async_greet", {"name": "world"})
        assert response["payload"]["result"] == "hello world"

    async def test_optional_arg_uses_default_when_omitted(self):
        session = _make_disconnected_session(None)

        @tool(description="greets, loudly if asked")
        def greet(name: str, loudly: bool = False) -> str:
            return f"{name.upper()}!!" if loudly else f"hello {name}"

        session.register_tools([greet])

        response = await self._run_call(session, "greet", {"name": "sam"})
        assert response["payload"]["result"] == "hello sam"

    async def test_unregistered_tool_returns_error_string(self):
        session = _make_disconnected_session(None)

        response = await self._run_call(session, "does_not_exist", {})
        assert "not registered on client" in response["payload"]["result"]

    async def test_tool_that_raises_returns_error_string_and_invokes_on_error(self):
        session = _make_disconnected_session(None)

        on_error = MagicMock()
        session.on_error = on_error

        @tool(description="always fails")
        def flaky() -> str:
            raise RuntimeError("boom")

        session.register_tools([flaky])

        response = await self._run_call(session, "flaky", {})
        assert "Error executing tool 'flaky'" in response["payload"]["result"]
        assert "boom" in response["payload"]["result"]
        on_error.assert_called_once()

    async def test_missing_required_arg_returns_error_string(self):
        session = _make_disconnected_session(None)

        @tool(description="requires x")
        def needs_x(x: str) -> str:
            return x

        session.register_tools([needs_x])

        # Simulate the LLM calling without the required argument.
        response = await self._run_call(session, "needs_x", {})
        assert "Error executing tool 'needs_x'" in response["payload"]["result"]

    async def test_non_string_result_is_json_serialized(self):
        session = _make_disconnected_session(None)

        @tool(description="returns structured data")
        def get_status() -> dict:
            return {"ok": True, "count": 3}

        session.register_tools([get_status])

        response = await self._run_call(session, "get_status", {})
        assert response["payload"]["result"] == '{"ok": true, "count": 3}'

    async def test_result_not_sent_when_disconnected(self):
        session = _make_disconnected_session(None)
        session._is_connected = False
        session.websocket = AsyncMock()

        @tool(description="trivial")
        def fn() -> str:
            return "x"

        session.register_tools([fn])

        await session._handle_client_tool_call(
            {"name": "fn", "args": {}, "tool_call_id": "c1"}
        )
        session.websocket.send.assert_not_called()


class TestSerializeToolResult:
    def test_string_passthrough(self):
        session = _make_disconnected_session(None)
        assert session._serialize_tool_result("already a string") == "already a string"

    def test_dict_is_json_encoded(self):
        session = _make_disconnected_session(None)
        assert session._serialize_tool_result({"a": 1}) == '{"a": 1}'

    def test_list_is_json_encoded(self):
        session = _make_disconnected_session(None)
        assert session._serialize_tool_result([1, 2, 3]) == "[1, 2, 3]"

    def test_unencodable_object_falls_back_to_str(self):
        session = _make_disconnected_session(None)

        class Weird:
            def __str__(self):
                return "weird-repr"

        # default=str in json.dumps means most objects DO serialize via str();
        # this exercises that fallback path rather than a hard TypeError.
        result = session._serialize_tool_result(Weird())
        assert "weird-repr" in result


# ===========================================================================
# 5. AgentSession - end-to-end tests against a live agent (client/models fixtures)
# ===========================================================================
#
# These mirror the existing `test_client_tools` example: they require the
# `client` and `models` fixtures wired up to a real (or test-environment)
# deployment target, since they exercise the full deploy -> connect ->
# chat -> client_tool_call -> client_tool_result round trip.


async def test_client_tools_no_args(client, models):
    """Baseline: a single tool with no arguments (matches the existing example)."""
    tool_called = False

    @tool(description="Get the current time in UTC")
    def get_time() -> str:
        nonlocal tool_called
        tool_called = True
        return datetime.now(timezone.utc).isoformat()

    manifest = AgentManifest(
        id="TestAgentNoArgs",
        system_prompt=(
            "If the user asks you to call a tool, "
            "call it immediately without asking any additional questions, "
            "even if the tool requires arguments, make them up."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[get_time],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([get_time])

    result = await session.chat("What is the UTC time now?")

    assert isinstance(result, str)
    assert tool_called


async def test_client_tools_required_args(client, models):
    """A tool with a single required argument."""
    received: dict[str, str] = {}

    @tool(description="Look up the weather for a given city")
    def get_weather(city: str) -> str:
        received["city"] = city
        return f"It's sunny in {city}."

    manifest = AgentManifest(
        id="TestAgentRequiredArgs",
        system_prompt=(
            "If the user asks about the weather somewhere, call get_weather "
            "with that city immediately, without asking clarifying questions."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[get_weather],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([get_weather])

    result = await session.chat("What's the weather like in Lisbon?")

    assert isinstance(result, str)
    assert received.get("city")


async def test_client_tools_optional_args_default_used(client, models):
    """A tool with an optional argument the models may omit; the default
    should be applied by the local implementation when omitted."""
    calls: list[dict] = []

    @tool(description="Sets the playback volume. Level defaults to 50 if unspecified.")
    def set_volume(level: int = 50) -> str:
        calls.append({"level": level})
        return f"Volume set to {level}"

    manifest = AgentManifest(
        id="TestAgentOptionalArgs",
        system_prompt=(
            "If the user asks you to turn the volume up without giving a "
            "specific number, call set_volume with no arguments so the "
            "default is used."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[set_volume],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([set_volume])

    result = await session.chat("Turn the volume up, I don't care to what level.")

    assert isinstance(result, str)
    assert len(calls) >= 1


async def test_client_tools_multiple_mixed_args(client, models):
    """A tool combining required and optional arguments of different types."""
    bookings: list[dict] = []

    @tool(description="Books a flight for a number of passengers, optionally refundable")
    def book_flight(destination: str, passengers: int, refundable: bool = False) -> str:
        bookings.append(
            {"destination": destination, "passengers": passengers, "refundable": refundable}
        )
        return f"Booked {passengers} seat(s) to {destination} (refundable={refundable})"

    manifest = AgentManifest(
        id="TestAgentMixedArgs",
        system_prompt=(
            "If the user asks to book a flight, call book_flight immediately "
            "with the destination and number of passengers they mention. "
            "Make up any detail they didn't specify."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[book_flight],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([book_flight])

    result = await session.chat("Book a flight to Berlin for 3 people.")

    assert isinstance(result, str)
    assert bookings
    assert bookings[0]["destination"]
    assert bookings[0]["passengers"] == 3


async def test_client_toolbelt(client, models):
    """A ``ToolBelt`` with several tools registered at once."""
    belt = ToolBelt()
    calls: dict[str, int] = {"get_time": 0, "get_battery_level": 0}

    @belt.tool(description="Get the current time in UTC")
    def get_time() -> str:
        calls["get_time"] += 1
        return datetime.now(timezone.utc).isoformat()

    @belt.tool(description="Get the device's current battery level as a percentage")
    def get_battery_level() -> str:
        calls["get_battery_level"] += 1
        return "87"

    manifest = AgentManifest(
        id="TestAgentToolBelt",
        system_prompt=(
            "If the user asks for the time or the battery level, call the "
            "matching tool immediately without asking any questions."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[belt],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([belt])

    result = await session.chat("What's the current battery level?")

    assert isinstance(result, str)
    assert calls["get_battery_level"] >= 1


async def test_client_async_tool(client, models):
    """An ``async def`` tool implementation."""
    tool_called = False

    @tool(description="Fetches the user's current location asynchronously")
    async def get_location() -> str:
        nonlocal tool_called
        await asyncio.sleep(0)
        tool_called = True
        return "Eygelshoven, NL"

    manifest = AgentManifest(
        id="TestAgentAsyncTool",
        system_prompt=(
            "If the user asks where they are, call get_location immediately."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[get_location],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([get_location])

    result = await session.chat("Where am I right now?")
    print(result)

    assert isinstance(result, str)
    assert tool_called


async def test_client_tool_declared_but_not_registered_still_completes(client, models):
    """The manifest declares a client tool, but the local implementation is
    never registered. The agent should still finish the turn (surfacing a
    'not registered on client' error to the LLM rather than hanging)."""

    @tool(description="Reads a secret only the client can access")
    def read_secret() -> str:
        return "should never actually run"

    manifest = AgentManifest(
        id="TestAgentUnregisteredTool",
        system_prompt=(
            "If the user asks for the secret, call read_secret immediately. "
            "If the tool call fails, just report that briefly and stop."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[read_secret],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    # Intentionally NOT calling session.register_tools([read_secret]) here.

    result = await session.chat("What is the secret?")
    print(result)

    assert isinstance(result, str)
    assert result  # the agent should still produce some final text


async def test_client_tool_execution_error_is_reported_and_recovered(client, models):
    """A tool implementation that raises should not crash the session; the
    error is sent back to the models as the tool result and the turn still
    completes."""
    attempts = 0
    on_error = MagicMock()

    @tool(description="A tool that always raises an error")
    def broken_tool() -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("simulated failure")

    manifest = AgentManifest(
        id="TestAgentToolError",
        system_prompt=(
            "If the user asks you to run the broken tool, call broken_tool "
            "immediately. If it errors, just tell the user briefly and stop."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[broken_tool],
    )

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([broken_tool])
    session.on_error = on_error

    result = await session.chat("Please run the broken tool.")

    assert isinstance(result, str)
    assert attempts >= 1


async def test_client_tool_broadcast_config_deploys_successfully(client, models):
    """A manifest declaring a broadcast-enabled client tool deploys and can
    complete a normal (non-broadcast, single-client) turn without error -
    broadcast aggregation itself requires multiple simultaneous connections
    and is covered at the config-validation level in
    ``TestBroadcastConfig`` above."""

    @tool(
        description="Asks every connected device to confirm an action",
        broadcast=BroadcastConfig(mode=BroadcastMode.collect, collect_window=1.0),
    )
    def confirm_on_all_devices() -> str:
        return "confirmed"

    manifest = AgentManifest(
        id="TestAgentBroadcastTool",
        system_prompt=(
            "If the user asks to confirm on all devices, call "
            "confirm_on_all_devices immediately."
        ),
        models=models,
        memory=MemoryConfig(enabled=False),
        tools=[confirm_on_all_devices],
    )

    assert manifest.tools[0].broadcast.mode.value == "collect"

    session = await client.aelite.get_session_from_manifest(manifest)
    session.register_tools([confirm_on_all_devices])

    result = await session.chat("Please confirm on all my devices.")
    assert isinstance(result, str)