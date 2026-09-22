"""
test_mcp.py - SDK-level tests for MCP server configuration.

Scope
-----
This file tests only what lives in the SDK itself (``slet_sdk.aelite.manifest``):
declaring and validating ``MCPServerConfig`` and its transport variants, and
how they integrate into ``AgentManifest.mcp_servers``. This mirrors the
scope of ``test_tools.py`` - pure Pydantic models behavior, no network, no
server-side dependencies.

Explicitly out of scope (server-side, not part of the SDK):
    - Actually loading/connecting to an MCP server (``mcp_service.py``:
      ``load_mcp_server``, ``resolve_mcp_call``, ``_read_resource``,
      ``fetch_prompt_value``, ``MCPServerData``).
    - The WS ``mcp_result`` round trip (``client_tools.py`` / the
      ``agents.py`` WS handler).
    - ``{mcp_res_*}`` / ``{mcp_prompt_*}`` placeholder rendering
      (``agent_factory.py::_render_template``).
    - MCP tools merging into ``inherit_tools_from`` / ``tool_retriever``.
These all require server-side infrastructure to exercise meaningfully and
belong in the server's own test suite, not the SDK's.

Per the task, SSE is only tested where needed for contrast (the
discriminated union needs both branches to prove ``type`` correctly
selects between them) - the real transport under test throughout is
stdio, since it's the one this SDK's client actually executes (the server
only ever receives ``server_id`` for stdio, per ``MCPServerConfig``'s
docstring), and it doesn't require a reachable remote server the way SSE
would for a genuine e2e test.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slet_sdk.aelite.manifest import (
    AgentManifest,
    MCPResourceMode,
    MCPServerConfig,
    MCPTransportSSE,
    MCPTransportStdio,
)


# ===========================================================================
# 1. MCPTransportStdio - unit tests
# ===========================================================================


class TestMCPTransportStdio:
    def test_command_is_required(self):
        with pytest.raises(ValidationError, match="command is required"):
            MCPTransportStdio()

    def test_command_none_explicitly_raises(self):
        with pytest.raises(ValidationError, match="command is required"):
            MCPTransportStdio(command=None)

    def test_empty_string_command_raises(self):
        # falsy but not None - the validator checks `if not self.command`,
        # so an empty string should be rejected the same as None.
        with pytest.raises(ValidationError, match="command is required"):
            MCPTransportStdio(command="")

    def test_minimal_valid_stdio_transport(self):
        t = MCPTransportStdio(command="npx")
        assert t.type == "stdio"
        assert t.command == "npx"
        assert t.args == []
        assert t.env == {}

    def test_args_and_env_defaults_are_independent_lists_dicts(self):
        """Guards against a shared mutable default across instances."""
        t1 = MCPTransportStdio(command="npx")
        t2 = MCPTransportStdio(command="uvx")

        t1.args.append("--foo")
        t1.env["KEY"] = "value"

        assert t2.args == []
        assert t2.env == {}

    def test_full_stdio_transport(self):
        t = MCPTransportStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            env={"NODE_ENV": "production"},
        )
        assert t.command == "npx"
        assert t.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        assert t.env == {"NODE_ENV": "production"}

    def test_type_field_is_fixed_literal(self):
        t = MCPTransportStdio(command="npx")
        assert t.type == "stdio"
        # The literal is a documentation/discriminator field, not meant to
        # be overridden - passing a mismatched type should fail validation
        # rather than silently accept it.
        with pytest.raises(ValidationError):
            MCPTransportStdio(command="npx", type="sse")

    def test_command_is_never_transmitted_marker_present_in_class_docs(self):
        """
        Not a runtime behavior test (the SDK has no transmission logic to
        test - that's server-side), just a guard that the documented
        security property stays attached to MCPServerConfig's docstring,
        so a future refactor doesn't silently drop the explanation of
        *why* `command` is required despite doing nothing at request time
        from the SDK's perspective for stdio transports.

        Note: this property is documented on MCPServerConfig's class
        docstring (describing the stdio transport as a whole) - NOT on
        MCPTransportStdio.command's own field description, which only
        explains that the server identifies the server by `id`.
        """
        assert "never transmitted" in MCPServerConfig.__doc__


# ===========================================================================
# 2. MCPTransportSSE - contrast only, for the discriminated union
# ===========================================================================


class TestMCPTransportSSEContrast:
    """
    SSE itself isn't under test here (that needs a reachable remote server
    for a meaningful e2e check, which we're explicitly not doing). This
    class only proves the discriminated union in MCPServerConfig.transport
    correctly separates the two transport kinds by `type`, which stdio
    tests alone can't demonstrate.
    """

    def test_url_is_required(self):
        with pytest.raises(ValidationError, match="url is required"):
            MCPTransportSSE()

    def test_minimal_valid_sse_transport(self):
        t = MCPTransportSSE(url="https://example.com/mcp")
        assert t.type == "sse"
        assert t.headers == {}

    def test_headers_default_is_independent_dict(self):
        t1 = MCPTransportSSE(url="https://a.example.com")
        t2 = MCPTransportSSE(url="https://b.example.com")
        t1.headers["Authorization"] = "Bearer x"
        assert t2.headers == {}


# ===========================================================================
# 3. MCPServerConfig - unit tests (stdio-focused)
# ===========================================================================


class TestMCPServerConfig:
    def test_id_is_required(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(transport=MCPTransportStdio(command="npx"))

    def test_minimal_stdio_server_config_defaults(self):
        cfg = MCPServerConfig(
            id="fs",
            transport=MCPTransportStdio(command="npx", args=["-y", "server-filesystem"]),
        )
        assert cfg.id == "fs"
        assert isinstance(cfg.transport, MCPTransportStdio)
        assert cfg.transport.command == "npx"

        # Defaults per docstring
        assert cfg.load_tools is True
        assert cfg.load_resources is False
        assert cfg.load_prompts is False
        assert cfg.resource_mode == MCPResourceMode.tool
        assert cfg.timeout == 30

    def test_default_transport_is_sse_when_omitted(self):
        """
        MCPServerConfig.transport defaults to MCPTransportSSE() via
        default_factory when omitted entirely - but MCPTransportSSE()
        itself requires `url`, so omitting transport with no override at
        all should fail validation rather than silently produce an
        unusable SSE config.
        """
        with pytest.raises(ValidationError):
            MCPServerConfig(id="fs")

    def test_transport_discriminator_selects_stdio(self):
        cfg = MCPServerConfig(
            id="fs",
            transport={"type": "stdio", "command": "npx"},
        )
        assert isinstance(cfg.transport, MCPTransportStdio)
        assert cfg.transport.command == "npx"

    def test_transport_discriminator_selects_sse(self):
        cfg = MCPServerConfig(
            id="remote",
            transport={"type": "sse", "url": "https://example.com/mcp"},
        )
        assert isinstance(cfg.transport, MCPTransportSSE)
        assert cfg.transport.url == "https://example.com/mcp"

    def test_transport_discriminator_rejects_unknown_type(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(id="fs", transport={"type": "websocket", "command": "npx"})

    def test_stdio_transport_missing_command_surfaces_through_server_config(self):
        with pytest.raises(ValidationError, match="command is required"):
            MCPServerConfig(id="fs", transport={"type": "stdio"})

    def test_timeout_must_be_at_least_one(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(
                id="fs",
                transport=MCPTransportStdio(command="npx"),
                timeout=0,
            )

    def test_custom_timeout_accepted(self):
        cfg = MCPServerConfig(
            id="fs",
            transport=MCPTransportStdio(command="npx"),
            timeout=90,
        )
        assert cfg.timeout == 90

    def test_load_resources_and_prompts_can_be_enabled(self):
        cfg = MCPServerConfig(
            id="fs",
            transport=MCPTransportStdio(command="npx"),
            load_resources=True,
            load_prompts=True,
            resource_mode=MCPResourceMode.system_prompt,
        )
        assert cfg.load_resources is True
        assert cfg.load_prompts is True
        assert cfg.resource_mode == MCPResourceMode.system_prompt

    def test_load_tools_can_be_disabled(self):
        """
        A server attached purely for resources/prompts, with tool loading
        turned off - should be representable without contradiction.
        """
        cfg = MCPServerConfig(
            id="docs",
            transport=MCPTransportStdio(command="npx"),
            load_tools=False,
            load_resources=True,
        )
        assert cfg.load_tools is False
        assert cfg.load_resources is True


class TestMCPResourceMode:
    def test_all_documented_modes_exist(self):
        assert set(MCPResourceMode) == {
            MCPResourceMode.tool,
            MCPResourceMode.tool_with_retriever,
            MCPResourceMode.system_prompt,
            MCPResourceMode.message,
        }

    def test_values_are_plain_strings(self):
        """MCPResourceMode is a Enum - values should compare equal to
        their string form (relevant since resource_mode ends up in
        serialized manifests and placeholder-building logic elsewhere)."""
        assert MCPResourceMode.tool == "tool"
        assert MCPResourceMode.tool_with_retriever == "tool_with_retriever"
        assert MCPResourceMode.system_prompt == "system_prompt"
        assert MCPResourceMode.message == "message"

    @pytest.mark.parametrize(
        "mode",
        [
            MCPResourceMode.tool,
            MCPResourceMode.tool_with_retriever,
            MCPResourceMode.system_prompt,
            MCPResourceMode.message,
        ],
    )
    def test_each_mode_accepted_on_server_config(self, mode):
        cfg = MCPServerConfig(
            id="fs",
            transport=MCPTransportStdio(command="npx"),
            load_resources=True,
            resource_mode=mode,
        )
        assert cfg.resource_mode == mode

    def test_invalid_resource_mode_rejected(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(
                id="fs",
                transport=MCPTransportStdio(command="npx"),
                resource_mode="not_a_real_mode",
            )


# ===========================================================================
# 4. Integration into AgentManifest.mcp_servers
# ===========================================================================


class TestAgentManifestMCPServers:
    def test_mcp_servers_defaults_to_empty_list(self):
        manifest = AgentManifest(id="A")
        assert manifest.mcp_servers == []

    def test_single_stdio_server_attached(self):
        manifest = AgentManifest(
            id="A",
            mcp_servers=[
                MCPServerConfig(
                    id="fs",
                    transport=MCPTransportStdio(
                        command="npx",
                        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                    ),
                )
            ],
        )
        assert len(manifest.mcp_servers) == 1
        assert manifest.mcp_servers[0].id == "fs"
        assert isinstance(manifest.mcp_servers[0].transport, MCPTransportStdio)

    def test_multiple_stdio_servers_with_distinct_ids(self):
        manifest = AgentManifest(
            id="A",
            mcp_servers=[
                MCPServerConfig(id="fs", transport=MCPTransportStdio(command="npx")),
                MCPServerConfig(id="git", transport=MCPTransportStdio(command="uvx")),
            ],
        )
        ids = {s.id for s in manifest.mcp_servers}
        assert ids == {"fs", "git"}

    def test_mcp_servers_accepts_plain_dicts(self):
        """
        Manifests are frequently built from JSON (deployed via
        POST /agent/deploy/{session_id}), so dict input for mcp_servers
        entries must work the same as constructing MCPServerConfig
        directly.
        """
        manifest = AgentManifest(
            id="A",
            mcp_servers=[
                {
                    "id": "fs",
                    "transport": {"type": "stdio", "command": "npx", "args": ["-y", "server-fs"]},
                }
            ],
        )
        assert isinstance(manifest.mcp_servers[0], MCPServerConfig)
        assert isinstance(manifest.mcp_servers[0].transport, MCPTransportStdio)
        assert manifest.mcp_servers[0].transport.args == ["-y", "server-fs"]

    def test_manifest_does_not_deduplicate_or_validate_duplicate_server_ids(self):
        """
        Unlike tool names (which AgentManifest enforces as unique across
        the flattened `tools` list), MCPServerConfig has no equivalent
        uniqueness validator on `mcp_servers` in the SDK today. This test
        documents current behavior rather than asserting it's desirable -
        duplicate server ids are a server-side concern (name-collision
        prefixing for generated tool names, per MCPServerConfig.id's
        docstring) and if uniqueness should be enforced, that validator
        belongs in this same place tool-name uniqueness already lives
        (AgentManifest.validate_timeouts_and_tools).
        """
        manifest = AgentManifest(
            id="A",
            mcp_servers=[
                MCPServerConfig(id="fs", transport=MCPTransportStdio(command="npx")),
                MCPServerConfig(id="fs", transport=MCPTransportStdio(command="uvx")),
            ],
        )
        assert len(manifest.mcp_servers) == 2
        assert all(s.id == "fs" for s in manifest.mcp_servers)

    def test_manifest_round_trips_stdio_server_through_json(self):
        """
        Manifests are serialized to Redis via model_dump_json() and
        restored via model_validate_json() (see manifest.py's own
        docstring). A stdio MCP server must survive that round trip
        intact, including nested transport fields.
        """
        original = AgentManifest(
            id="A",
            mcp_servers=[
                MCPServerConfig(
                    id="fs",
                    transport=MCPTransportStdio(
                        command="npx",
                        args=["-y", "server-fs"],
                        env={"NODE_ENV": "production"},
                    ),
                    load_resources=True,
                    resource_mode=MCPResourceMode.message,
                    timeout=45,
                )
            ],
        )

        restored = AgentManifest.model_validate_json(original.model_dump_json())

        assert len(restored.mcp_servers) == 1
        srv = restored.mcp_servers[0]
        assert srv.id == "fs"
        assert isinstance(srv.transport, MCPTransportStdio)
        assert srv.transport.command == "npx"
        assert srv.transport.args == ["-y", "server-fs"]
        assert srv.transport.env == {"NODE_ENV": "production"}
        assert srv.load_resources is True
        assert srv.resource_mode == MCPResourceMode.message
        assert srv.timeout == 45

    def test_manifest_with_mcp_servers_and_no_tools_still_valid(self):
        """An agent may rely entirely on MCP-provided tools, with no
        declarative `tools` of its own."""
        manifest = AgentManifest(
            id="A",
            tools=[],
            mcp_servers=[MCPServerConfig(id="fs", transport=MCPTransportStdio(command="npx"))],
        )
        assert manifest.tools == []
        assert len(manifest.mcp_servers) == 1

    def test_sub_agent_can_declare_its_own_mcp_servers(self):
        """
        mcp_servers is per-manifest, same as tools - a sub-agent's own
        mcp_servers list is independent of its parent's, mirroring how
        `tools` works for sub_agents (see AgentManifest.sub_agents'
        docstring: sub-agents are fully recursive, independently deployed
        AgentManifest instances).
        """
        child = AgentManifest(
            id="child",
            mcp_servers=[MCPServerConfig(id="git", transport=MCPTransportStdio(command="uvx"))],
        )
        parent = AgentManifest(
            id="parent",
            mcp_servers=[MCPServerConfig(id="fs", transport=MCPTransportStdio(command="npx"))],
            sub_agents=[child],
        )
        assert parent.mcp_servers[0].id == "fs"
        assert parent.sub_agents[0].mcp_servers[0].id == "git"