"""
test_memory_layers.py - end-to-end tests proving the three memory layers
(checkpointing / summarization / A-MEM) are independently togglable, in
any structurally valid combination.

Structural constraint under test
---------------------------------
`checkpointing` and `agentic` (A-MEM) are genuinely independent - A-MEM
extracts a note from the current exchange, which is available to the graph
regardless of whether the checkpointer is attached. `summarization`,
however, physically requires `checkpointing=True`: there is no persisted
history to trim/summarize otherwise. That single dependency is enforced by
`MemoryConfig.validate_memory_combination` and is tested at the config
level (no server needed) rather than end-to-end.

Requires the same live server + LLM chain + local/external embeddings
provider as test_amem.py (see its module docstring for `embeds` fixture
details).
"""

from __future__ import annotations

import pytest

from slet_sdk.aelite.manifest import (
    AgentManifest,
    AMEMConfig,
    EmbedConfig,
    MemoryConfig,
    ModelConfig,
)


def _amem_models(models: list[ModelConfig]) -> list[ModelConfig]:
    marked = [m.model_copy(deep=True) for m in models]
    for m in marked:
        m.use_for_amem = True
    return marked


_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n\n"
    "Relevant memory notes from prior conversations:\n{amem}\n\n"
    "Conversation summary so far:\n{summary}\n\n"
    "If the user shares a personal fact, acknowledge it in one short "
    "sentence - don't ask follow-up questions. If asked to recall "
    "something and you don't have it (in this conversation, the summary, "
    "or the memory notes above), say plainly that you don't have it."
)


def _manifest(
    agent_id: str,
    models: list[ModelConfig],
    *,
    checkpointing: bool,
    summarization: str = "disabled",
    amem_enabled: bool = False,
    embeds: list[EmbedConfig] | None = None,
    threshold: int = 6,
    keep_last: int = 3,
) -> AgentManifest:
    return AgentManifest(
        id=agent_id,
        system_prompt=_SYSTEM_PROMPT,
        models=_amem_models(models) if amem_enabled else models,
        embeds=embeds or [],
        memory=MemoryConfig(
            checkpointing=checkpointing,
            summarization=summarization,
            threshold=threshold,
            keep_last=keep_last,
            agentic=AMEMConfig(
                enabled=amem_enabled,
                scope="session",
                write_mode="blocking",
            ),
        ),
    )


# ===========================================================================
# 1. checkpointing alone ("regular memory"), no summary, no A-MEM
# ===========================================================================


async def test_checkpointing_alone_gives_continuity_within_thread(client, models):
    """checkpointing=True, summarization='disabled', A-MEM off: the model
    still sees earlier turns in the same thread via the persisted
    checkpoint, without any summarization or A-MEM machinery involved."""
    manifest = _manifest("MemCheckpointOnly", models, checkpointing=True)
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    await session.chat("My favorite color is teal. Just acknowledge that.")
    result = await session.chat("What's my favorite color?")

    assert isinstance(result, str)
    assert "teal" in result.lower()


async def test_checkpointing_disabled_gives_no_continuity(client, models):
    """checkpointing=False: every turn starts from a clean graph state, so
    a fact from a previous turn in the *same* thread is not visible - the
    only thing that could make it visible is A-MEM, which is off here."""
    manifest = _manifest("MemNoCheckpoint", models, checkpointing=False)
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    await session.chat("My favorite color is teal. Just acknowledge that.")
    result = await session.chat("What's my favorite color?")

    assert isinstance(result, str)
    assert "teal" not in result.lower()


# ===========================================================================
# 2. A-MEM alone, checkpointing OFF - the key new capability
# ===========================================================================


async def test_amem_alone_without_checkpointing_still_recalls(client, models, embeds):
    """
    checkpointing=False, agentic.enabled=True: A-MEM must still work. Its
    note extraction only needs the current exchange (Human + AI messages
    from *this* graph invocation), which is available regardless of
    whether the checkpointer is attached - so cross-turn recall still
    works, but purely through A-MEM notes rather than persisted state.
    """
    manifest = _manifest(
        "MemAmemOnly", models,
        checkpointing=False,
        amem_enabled=True,
        embeds=embeds,
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session_a = await client.aelite.connect(thread.id)
    await session_a.chat(
        "Please remember this: my favorite programming language is Rust."
    )

    # A fresh connection - with checkpointing off, the graph has no
    # persisted history at all going into this turn. Anything recalled
    # here can only have come from A-MEM.
    session_b = await client.aelite.connect(thread.id)
    result = await session_b.chat(
        "What did I tell you my favorite programming language is?"
    )

    assert isinstance(result, str)
    assert "rust" in result.lower()


async def test_neither_layer_enabled_recalls_nothing(client, models, embeds):
    """checkpointing=False, agentic.enabled=False: no channel exists for a
    fact to survive between turns at all."""
    manifest = _manifest(
        "MemNeitherLayer", models,
        checkpointing=False,
        amem_enabled=False,
        embeds=embeds,
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session_a = await client.aelite.connect(thread.id)
    await session_a.chat("Please remember this: my lucky number is 42.")

    session_b = await client.aelite.connect(thread.id)
    result = await session_b.chat("What is my lucky number?")

    assert isinstance(result, str)
    assert "42" not in result


# ===========================================================================
# 3. All three layers together
# ===========================================================================


async def test_all_three_layers_enabled_together(client, models, embeds):
    """checkpointing=True, summarization='blocking', agentic.enabled=True:
    the three layers must coexist without interfering with each other."""
    manifest = _manifest(
        "MemAllThree", models,
        checkpointing=True,
        summarization="blocking",
        amem_enabled=True,
        embeds=embeds,
        threshold=2,
        keep_last=1,
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session_a = await client.aelite.connect(thread.id)
    await session_a.chat("Please remember this: my office is in Berlin.")
    # A couple more turns to push threshold/keep_last and trigger trimming
    # + summarization, exercising the post-trim A-MEM write path too.
    await session_a.chat("What's 2 + 2?")
    await session_a.chat("What's the capital of France?")

    session_b = await client.aelite.connect(thread.id)
    result = await session_b.chat("Where is my office?")

    assert isinstance(result, str)
    assert "berlin" in result.lower()


# ===========================================================================
# 4. Config-level validation of the one real structural dependency
# ===========================================================================


class TestMemoryCombinationValidation:
    def test_summarization_without_checkpointing_raises(self):
        with pytest.raises(ValueError, match="requires memory.checkpointing"):
            MemoryConfig(checkpointing=False, summarization="async")

    def test_summarization_disabled_without_checkpointing_is_valid(self):
        cfg = MemoryConfig(checkpointing=False, summarization="disabled")
        assert cfg.checkpointing is False

    def test_agentic_enabled_without_checkpointing_is_valid(self):
        """The core fix under test: A-MEM no longer requires checkpointing."""
        cfg = MemoryConfig(
            checkpointing=False,
            agentic=AMEMConfig(enabled=True, scope="session"),
        )
        assert cfg.agentic.enabled is True
        assert cfg.checkpointing is False

    def test_all_three_enabled_is_valid(self):
        cfg = MemoryConfig(
            checkpointing=True,
            summarization="async",
            agentic=AMEMConfig(enabled=True, scope="session"),
        )
        assert cfg.checkpointing and cfg.summarization == "async" and cfg.agentic.enabled

    def test_keep_last_must_be_below_threshold_still_enforced(self):
        with pytest.raises(ValueError, match="strictly less"):
            MemoryConfig(checkpointing=True, threshold=3, keep_last=3)