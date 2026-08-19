"""
test_amem.py - end-to-end tests for the A-MEM knowledge-graph feature,
exercised entirely through the public SDK (create thread -> connect ->
chat), the same way test_tools.py exercises client tools.

Scope
-----
These tests require a live server, a real (or test-environment) LLM chain,
and a real embeddings provider, since A-MEM's write path does an LLM note
extraction call and an embedding call, and its read path does a vector
search. They are deliberately end-to-end rather than unit tests - the SDK
itself has no A-MEM logic to unit test (all of it lives server-side in
``amem_service.py`` / ``agent_factory.py``); from the SDK's perspective
A-MEM is just config on ``AgentManifest.memory.agentic`` plus observable
chat behaviour.

``write_mode='blocking'`` is used throughout instead of the default
``'async'`` - it's the only way to make a test deterministic without
polling/sleeping for a fire-and-forget background task to land before the
next assertion.

Threads are created directly from a bare manifest via
``client.aelite.threads.create_thread(manifest=manifest)`` rather than
going through ``agents.create_agent()`` first, except where the test is
specifically about ``scope='agent'`` (which requires a real owning agent
template, i.e. ``thread.source_agent_id`` set) - those go through
``agents.create_agent()`` and then ``threads.create_thread(agent=agent.id)``.

``create_thread``/``create_agent`` are keyword-only past ``self`` in the
actual implementation (the ``@overload`` signatures only describe types for
the type checker) and are ``async`` - both the keyword and the ``await``
are required, or you get a ``TypeError``/silently-unawaited-coroutine bug
respectively.

Not covered here (would need SDK support this suite doesn't have access
to): sub-agent-level ``session`` vs ``thread`` scope distinction. With no
sub-agents, a manifest's own thread_id *is* its root_thread_id, so
``scope='session'`` and ``scope='thread'`` are namespace-equivalent for a
single, non-nested agent - the distinction only becomes observable across
a supervisor/sub-agent tree, which isn't exercised here.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from tests.helpers.auth import get_auth_data, get_client
from slet_sdk.aelite.manifest import (
    AgentManifest,
    AMEMConfig,
    EmbedConfig,
    MemoryConfig,
    ModelConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
async def other_client():
    """
    A second, independently signed-up user - needed for the group_user
    isolation test, which must prove two *different* users sharing the
    same group_key stay isolated when the scope has the `_user` suffix.
    """
    name, email, password = get_auth_data()

    async with get_client() as c:
        await c.signup(name, email, password)
        await c.signin(email, password)
        yield c


def _amem_models(models: list[ModelConfig]) -> list[ModelConfig]:
    """Returns a copy of `models` with every entry marked use_for_amem=True."""
    marked = [m.model_copy(deep=True) for m in models]
    for m in marked:
        m.use_for_amem = True
    return marked


_MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant with persistent memory of past "
    "conversations.\n\n"
    "Relevant memory notes from prior conversations:\n{amem}\n\n"
    "If the user shares a personal fact (a preference, a name, a detail "
    "about themselves), acknowledge it in one short sentence - don't ask "
    "follow-up questions. If the user asks you to recall something and a "
    "relevant note is present above, state that fact plainly. If no "
    "relevant note is present, say you don't have that information yet."
)


def _amem_manifest(
    agent_id: str,
    models: list[ModelConfig],
    embeds: list[EmbedConfig],
    amem_cfg: AMEMConfig,
) -> AgentManifest:
    return AgentManifest(
        id=agent_id,
        system_prompt=_MEMORY_SYSTEM_PROMPT,
        models=_amem_models(models),
        embeds=embeds,
        memory=MemoryConfig(checkpointing=True, agentic=amem_cfg),
    )


# ===========================================================================
# 1. Basic persistence within a single scope
# ===========================================================================


async def test_amem_session_scope_persists_across_reconnects(client, models, embeds):
    """
    scope='session': a fact told in one connection to a thread should be
    recallable after disconnecting and reconnecting to the *same* thread -
    proving notes survive independently of any single WebSocket connection
    or in-process graph cache entry.
    """
    amem_cfg = AMEMConfig(
        enabled=True,
        scope="session",
        write_mode="blocking",
        retrieval_top_k=5,
    )
    manifest = _amem_manifest("AmemSessionAgent", models, embeds, amem_cfg)

    thread = await client.aelite.threads.create_thread(manifest=manifest)

    first_session = await client.aelite.connect(thread.id)
    await first_session.chat(
        "Please remember this for later: my favorite programming language is Rust."
    )

    second_session = await client.aelite.connect(thread.id)
    result = await second_session.chat(
        "What did I tell you my favorite programming language is?"
    )

    assert isinstance(result, str)
    assert "rust" in result.lower()


async def test_amem_session_scope_isolated_between_threads(client, models, embeds):
    """
    scope='session': two independent threads must NOT share notes - each
    thread's memory is private to itself, even with an identical manifest.
    """
    amem_cfg = AMEMConfig(
        enabled=True,
        scope="session",
        write_mode="blocking",
    )
    manifest = _amem_manifest("AmemSessionIsolationAgent", models, embeds, amem_cfg)

    thread_a = await client.aelite.threads.create_thread(manifest=manifest)
    thread_b = await client.aelite.threads.create_thread(manifest=manifest)

    session_a = await client.aelite.connect(thread_a.id)
    await session_a.chat(
        "Please remember this: my secret code word is 'zeppelin'."
    )

    session_b = await client.aelite.connect(thread_b.id)
    result = await session_b.chat("What is my secret code word?")

    assert isinstance(result, str)
    assert "zeppelin" not in result.lower()


# ===========================================================================
# 2. Master switch
# ===========================================================================


async def test_amem_disabled_never_recalls(client, models, embeds):
    """enabled=False: no notes are ever written, so nothing is ever recalled,
    even in a fresh thread from the same agent."""
    amem_cfg = AMEMConfig(enabled=False)
    manifest = _amem_manifest("AmemDisabledAgent", models, embeds, amem_cfg)

    thread_a = await client.aelite.threads.create_thread(manifest=manifest)
    session_a = await client.aelite.connect(thread_a.id)
    await session_a.chat(
        "Please remember this: my secret code word is 'narwhal'."
    )

    # A new, unrelated thread - the only way it could know about
    # 'narwhal' is via A-MEM recall, which must not happen since
    # enabled=False. Reusing thread_a here would let the model see
    # 'narwhal' through the thread's own message history/checkpointer,
    # independently of A-MEM, and the assertion would pass for the
    # wrong reason.
    thread_b = await client.aelite.threads.create_thread(manifest=manifest)
    session_b = await client.aelite.connect(thread_b.id)
    result = await session_b.chat("What is my secret code word?")

    assert isinstance(result, str)
    assert "narwhal" not in result.lower()


# ===========================================================================
# 3. Agent scope - shared across threads of the same source agent
# ===========================================================================


async def test_amem_agent_scope_shared_across_sibling_threads(client, models, embeds):
    """
    scope='agent': two different threads deployed from the same agent
    template must share the same A-MEM namespace, since the namespace is
    keyed by the thread's source_agent_id, not the thread_id itself.
    """
    amem_cfg = AMEMConfig(
        enabled=True,
        scope="agent",
        write_mode="blocking",
    )
    manifest = _amem_manifest("AmemAgentScopeAgent", models, embeds, amem_cfg)
    agent = await client.aelite.agents.create_agent(manifest)

    thread_a = await client.aelite.threads.create_thread(agent=agent.id)
    thread_b = await client.aelite.threads.create_thread(agent=agent.id)

    session_a = await client.aelite.connect(thread_a.id)
    await session_a.chat(
        "Please remember this: my favorite city to visit is Lisbon."
    )

    session_b = await client.aelite.connect(thread_b.id)
    result = await session_b.chat("What is my favorite city to visit?")

    assert isinstance(result, str)
    assert "lisbon" in result.lower()


async def test_amem_agent_scope_isolated_across_different_agents(client, models, embeds):
    """
    scope='agent': two threads deployed from *different* agent templates
    must not share notes, even with identical config, since the namespace
    is keyed by source_agent_id.
    """
    amem_cfg = AMEMConfig(enabled=True, scope="agent", write_mode="blocking")

    manifest_1 = _amem_manifest("AmemAgentScopeOne", models, embeds, amem_cfg)
    manifest_2 = _amem_manifest("AmemAgentScopeTwo", models, embeds, amem_cfg)

    agent_1 = await client.aelite.agents.create_agent(manifest_1)
    agent_2 = await client.aelite.agents.create_agent(manifest_2)

    thread_1 = await client.aelite.threads.create_thread(agent=agent_1.id)
    thread_2 = await client.aelite.threads.create_thread(agent=agent_2.id)

    session_1 = await client.aelite.connect(thread_1.id)
    await session_1.chat("Please remember this: my pet's name is Biscuit.")

    session_2 = await client.aelite.connect(thread_2.id)
    result = await session_2.chat("What is my pet's name?")

    assert isinstance(result, str)
    assert "biscuit" not in result.lower()


async def test_amem_agent_scope_without_source_agent_reports_error_but_still_replies(
    client, models, embeds
):
    """
    scope='agent' on a thread created directly from a bare manifest (no
    owning agent template via create_thread(agent=...), so
    thread.source_agent_id is None) does not crash or hang the turn -
    _resolve_namespace's ValueError is caught in amem_read_node/call_model
    and reported out-of-band via _notify_amem_error(thread_id, e), which
    calls send_msg_error with msg_id=None.

    Since msg_id=None never matches this chat() call's own response queue
    (AgentSession._route_error only routes into a waiter's queue when
    msg_id is truthy - see `queue = self._response_queues.get(msg_id) if
    msg_id else None`), the error surfaces exclusively through the global
    on_server_error callback. chat() itself completes normally with the
    model's actual reply, just without any memory context injected.
    """
    amem_cfg = AMEMConfig(enabled=True, scope="agent", write_mode="blocking")
    manifest = _amem_manifest("AmemAgentScopeNoTemplate", models, embeds, amem_cfg)

    # Created directly from a manifest, with no owning agent - no
    # source_agent_id, so scope='agent' has nothing to key its namespace on.
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)

    server_errors: list[tuple] = []
    session.on_server_error = lambda error, msg_id: server_errors.append((error, msg_id))

    result = await session.chat("Hello there, please just say hi back.")

    assert isinstance(result, str)
    assert result  # the turn completed normally despite the A-MEM failure

    # on_server_error is invoked via asyncio.create_task() from a sync
    # dispatch path (_invoke_callback), so chat() returning doesn't
    # guarantee the callback has actually run yet - give the event loop a
    # beat before asserting on it.
    await asyncio.sleep(0.1)

    assert len(server_errors) == 1
    error, msg_id = server_errors[0]
    assert msg_id is None


# ===========================================================================
# 4. Group scope - arbitrary shared key
# ===========================================================================


async def test_amem_group_scope_shared_via_group_key(client, models, embeds):
    """
    scope='group': two threads with no relationship at all (different
    manifests, no owning agent) share notes purely because they set the
    same group_key.
    """
    group_key = f"amem-e2e-{uuid.uuid4()}"
    amem_cfg = AMEMConfig(
        enabled=True,
        scope="group",
        group_key=group_key,
        write_mode="blocking",
    )

    manifest_1 = _amem_manifest("AmemGroupAgentOne", models, embeds, amem_cfg)
    manifest_2 = _amem_manifest("AmemGroupAgentTwo", models, embeds, amem_cfg)

    thread_1 = await client.aelite.threads.create_thread(manifest=manifest_1)
    thread_2 = await client.aelite.threads.create_thread(manifest=manifest_2)

    session_1 = await client.aelite.connect(thread_1.id)
    await session_1.chat(
        "Please remember this: the project codename is 'Blue Falcon'."
    )

    session_2 = await client.aelite.connect(thread_2.id)
    result = await session_2.chat("What is the project codename?")

    assert isinstance(result, str)
    assert "blue falcon" in result.lower()


async def test_amem_group_scope_isolated_without_matching_key(client, models, embeds):
    """
    Two threads with different group_key values must not share notes, even
    though both use scope='group' - membership is purely by exact key match.
    """
    amem_cfg_1 = AMEMConfig(
        enabled=True,
        scope="group",
        group_key=f"amem-e2e-{uuid.uuid4()}",
        write_mode="blocking",
    )
    amem_cfg_2 = AMEMConfig(
        enabled=True,
        scope="group",
        group_key=f"amem-e2e-{uuid.uuid4()}",
        write_mode="blocking",
    )

    manifest_1 = _amem_manifest("AmemGroupMismatchOne", models, embeds, amem_cfg_1)
    manifest_2 = _amem_manifest("AmemGroupMismatchTwo", models, embeds, amem_cfg_2)

    thread_1 = await client.aelite.threads.create_thread(manifest=manifest_1)
    thread_2 = await client.aelite.threads.create_thread(manifest=manifest_2)

    session_1 = await client.aelite.connect(thread_1.id)
    await session_1.chat("Please remember this: the vault combination is '17-42-9'.")

    session_2 = await client.aelite.connect(thread_2.id)
    result = await session_2.chat("What is the vault combination?")

    assert isinstance(result, str)
    assert "17-42-9" not in result


async def test_amem_group_user_scope_isolates_by_user(client, other_client, models, embeds):
    """
    scope='group_user': two different users, both setting the same
    group_key, must NOT see each other's notes - the `_user` suffix
    isolates on top of the shared key.
    """
    group_key = f"amem-e2e-user-{uuid.uuid4()}"
    amem_cfg = AMEMConfig(
        enabled=True,
        scope="group_user",
        group_key=group_key,
        write_mode="blocking",
    )
    manifest = _amem_manifest("AmemGroupUserAgent", models, embeds, amem_cfg)

    thread = await client.aelite.threads.create_thread(manifest=manifest)
    other_thread = await other_client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    await session.chat(
        "Please remember this: my confidential access code is 'Orion-7'."
    )

    other_session = await other_client.aelite.connect(other_thread.id)
    result = await other_session.chat("What is my confidential access code?")

    assert isinstance(result, str)
    assert "orion-7" not in result.lower()


# ===========================================================================
# 5. Config validation (pure Pydantic, no server round-trip needed - grouped
#    here since they're specifically about the A-MEM scope/group_key feature)
# ===========================================================================


class TestAMEMScopeValidation:
    def test_group_scope_without_group_key_raises(self):
        with pytest.raises(ValueError, match="requires group_key"):
            AMEMConfig(enabled=True, scope="group")

    def test_group_user_scope_without_group_key_raises(self):
        with pytest.raises(ValueError, match="requires group_key"):
            AMEMConfig(enabled=True, scope="group_user")

    def test_group_key_below_min_length_raises(self):
        with pytest.raises(ValueError):
            AMEMConfig(enabled=True, scope="group", group_key="short")

    def test_group_key_accepted_at_min_length(self):
        cfg = AMEMConfig(enabled=True, scope="group", group_key="12345678")
        assert cfg.group_key == "12345678"

    def test_non_group_scope_does_not_require_group_key(self):
        cfg = AMEMConfig(enabled=True, scope="session")
        assert cfg.group_key is None

    @pytest.mark.parametrize(
        "scope",
        [
            "session",
            "session_user",
            "thread",
            "thread_user",
            "agent",
            "agent_user",
        ],
    )
    def test_group_key_ignored_but_allowed_for_non_group_scopes(self, scope):
        """
        group_key has no effect outside group/group_user scopes, but
        setting it alongside another scope is not itself an error - only
        the reverse (group scope without a key) is validated.
        """
        # noinspection PyTypeChecker
        cfg = AMEMConfig(enabled=True, scope=scope, group_key="12345678")
        assert cfg.group_key == "12345678"