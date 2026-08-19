"""
test_rag.py - end-to-end tests for the RAG (Retrieval-Augmented
Generation) feature, exercised entirely through the public SDK, the same
way test_amem.py exercises A-MEM.

Scope
-----
These tests require a live server, a real (or test-environment) LLM
chain, and a real embeddings provider (RAG's `embeds` requirement mirrors
A-MEM's - see test_amem.py's module docstring). They additionally require
a Redis Stack instance reachable by both the test process and the server
(`RAG_REDIS_URL`), since RAGSource(backend='redis') is the only backend
exercised here - it needs no extra service beyond Redis, which the project
already depends on elsewhere (checkpointer_redis / common.redis).

Unlike A-MEM, RAG has no write path exposed through the agent itself -
there is no "remember this document" tool. Test data is seeded directly
into the vector store via tests/helpers/rag_redis.py, exactly as an
operator would provision a real collection out-of-band before pointing an
agent at it.

`reranker.type='cross_encoder'` tests require the `rerankers` fixture
(a local CrossEncoder-backed /rerank double, see
tests/helpers/reranker_server.py) - skipped if RERANKER_MODEL/
RERANKER_BASE_URL is not configured, mirroring how `embeds` skips.

`reranker.type='llm'` tests reuse the `models` fixture, marking every
entry `use_for_rerank=True` (mirrors `_amem_models` marking
`use_for_amem=True`).
"""

from __future__ import annotations

import uuid

import pytest

from tests.helpers.rag_redis import seed_redis_rag_source, drop_redis_rag_source
from slet_sdk.aelite.manifest import (
    AgentManifest,
    EmbedConfig,
    ModelConfig,
    RAGRerankerConfig,
    RAGConfig,
    RAGSource,
    RerankerConfig,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _rerank_models(models: list[ModelConfig]) -> list[ModelConfig]:
    """Returns a copy of `models` with every entry marked use_for_rerank=True -
    mirrors test_amem.py's _amem_models for use_for_amem."""
    marked = [m.model_copy(deep=True) for m in models]
    for m in marked:
        m.use_for_rerank = True
    return marked


_RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a private knowledge base.\n\n"
    "Relevant documents:\n{rag_docs}\n\n"
    "Answer strictly using the documents above. If the answer is not "
    "present in the documents, say plainly that you don't have that "
    "information - do not use outside knowledge."
)

_RAG_TOOL_SYSTEM_PROMPT = (
    "You are a helpful assistant. You have a 'docs_search' tool that "
    "searches a private knowledge base - use it whenever the user asks "
    "about something that might be in it. Answer strictly using what the "
    "tool returns; if it returns nothing relevant, say so."
)


@pytest.fixture
async def rag_source(rag_redis_urls, embeds, embed_seed_base_url):
    """
    A single seeded RAGSource(backend='redis') with a handful of documents
    about a fictional product, unique per test (random collection +
    namespace) so tests never see each other's data even when run in
    parallel.
    """
    collection = f"rag-e2e-{uuid.uuid4().hex[:12]}"
    namespace = f"ns-{uuid.uuid4().hex[:12]}"
    embed_cfg = embeds[0]

    documents = [
        "The Zenith X200 has a battery life of 40 hours on a single charge.",
        "The Zenith X200 ships in three colors: slate, ivory, and forest green.",
        "The Zenith X200's warranty covers manufacturing defects for 2 years.",
        "To reset the Zenith X200, hold the power button for 15 seconds.",
    ]

    await seed_redis_rag_source(
        redis_url=rag_redis_urls[0],
        collection=collection,
        namespace=namespace,
        documents=documents,
        embed_base_url=embed_seed_base_url,
        embed_model=embed_cfg.model,
        embed_api_key=embed_cfg.api_key,
    )

    try:
        yield RAGSource(
            name="docs",
            backend="redis",
            collection=collection,
            connection_url=rag_redis_urls[1],
            namespace=namespace,
            top_k=4,
        )
    finally:
        await drop_redis_rag_source(rag_redis_urls[0], collection)


def _rag_manifest(
    agent_id: str,
    models: list[ModelConfig],
    embeds: list[EmbedConfig],
    source: RAGSource,
    *,
    inject_as: str = "system_variable",
    query_mode: str = "last_message",
    reranker: RAGRerankerConfig | None = None,
    rerankers: list[RerankerConfig] | None = None,
) -> AgentManifest:
    source = source.model_copy(update={"inject_as": inject_as})
    system_prompt = (
        _RAG_TOOL_SYSTEM_PROMPT if inject_as == "tool" else _RAG_SYSTEM_PROMPT
    )
    return AgentManifest(
        id=agent_id,
        system_prompt=system_prompt.replace("{rag_docs}", "{rag_" + source.name + "}"),
        models=models,
        embeds=embeds,
        rerankers=rerankers or [],
        rag=RAGConfig(
            enabled=True,
            sources=[source],
            query_mode=query_mode,
            reranker=reranker or RAGRerankerConfig(type="none", top_n=3),
        ),
    )


# ===========================================================================
# 1. Basic retrieval - system_variable injection
# ===========================================================================


async def test_rag_system_variable_injects_relevant_chunks(client, models, embeds, rag_source):
    """
    inject_as='system_variable': the model should be able to answer a
    question strictly from the seeded documents, since retrieval happens
    automatically before every LLM call and the chunks are injected via
    {rag_<name>}.
    """
    manifest = _rag_manifest("RagSystemVarAgent", models, embeds, rag_source)
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" in result


async def test_rag_system_variable_no_hallucination_outside_docs(client, models, embeds, rag_source):
    """A question with no answer in the seeded documents should not be
    answered from the model's own knowledge - the system prompt instructs
    it to say so explicitly when the injected context doesn't cover it."""
    manifest = _rag_manifest("RagNoHallucinationAgent", models, embeds, rag_source)
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("What is the Zenith X200's screen resolution?")

    assert isinstance(result, str)
    # Not a strict hallucination-detector - just checks the model didn't
    # confidently invent a specific resolution figure like "1920x1080".
    assert "1920" not in result and "resolution" not in result.lower() or (
        "don't" in result.lower() or "not" in result.lower()
    )


# ===========================================================================
# 2. Tool injection
# ===========================================================================


async def test_rag_tool_injection_search_tool_answers(client, models, embeds, rag_source):
    """
    inject_as='tool': retrieval is not automatic - the agent must decide
    to call `<name>_search`. RAGConfig.query_mode is irrelevant here since
    it only governs the auto-retrieval path.
    """
    manifest = _rag_manifest(
        "RagToolAgent", models, embeds, rag_source, inject_as="tool"
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat(
        "What colors does the Zenith X200 come in? Use your search tool."
    )

    assert isinstance(result, str)
    assert any(c in result.lower() for c in ("slate", "ivory", "forest"))


# ===========================================================================
# 3. query_mode
# ===========================================================================


async def test_rag_query_mode_last_message_uses_current_turn(client, models, embeds, rag_source):
    """query_mode='last_message' (default): retrieval is keyed off the
    most recent HumanMessage, so a topic-relevant question retrieves
    relevant chunks even as the first turn in the thread."""
    manifest = _rag_manifest(
        "RagLastMessageAgent", models, embeds, rag_source, query_mode="last_message"
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How do I reset the Zenith X200?")

    assert isinstance(result, str)
    assert "15" in result or "power button" in result.lower()


# ===========================================================================
# 4. Reranker types
# ===========================================================================


async def test_rag_reranker_none_still_returns_answer(client, models, embeds, rag_source):
    """type='none': chunks are concatenated in source order and truncated
    to top_n - no relevance judging, but retrieval + injection must still
    work end-to-end."""
    manifest = _rag_manifest(
        "RagRerankerNoneAgent", models, embeds, rag_source,
        reranker=RAGRerankerConfig(type="none", top_n=4),
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" in result


async def test_rag_reranker_rrf_still_returns_answer(client, models, embeds, rag_source):
    """type='rrf': Reciprocal Rank Fusion - no external dependency, should
    work with just the seeded Redis source."""
    manifest = _rag_manifest(
        "RagRerankerRrfAgent", models, embeds, rag_source,
        reranker=RAGRerankerConfig(type="rrf", top_n=2),
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" in result


async def test_rag_reranker_cross_encoder_uses_rerankers_chain(
    client, models, embeds, rerankers, rag_source
):
    """
    type='cross_encoder': requires manifest.rerankers with at least one
    use_for_rag=True entry - exercises the full path through
    ResilientRerankerModel / the local Cohere-compatible reranker double.
    """
    manifest = _rag_manifest(
        "RagRerankerCrossEncoderAgent", models, embeds, rag_source,
        reranker=RAGRerankerConfig(type="cross_encoder", top_n=2),
        rerankers=rerankers,
    )
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" in result


async def test_rag_reranker_llm_uses_use_for_rerank_models(client, models, embeds, rag_source):
    """
    type='llm': requires at least one entry in `models` with
    use_for_rerank=True - exercises the primary LLM acting as its own
    relevance judge.
    """
    for m in models:
        m.use_for_rerank = True
    manifest = _rag_manifest(
        "RagRerankerLlmAgent", models, embeds, rag_source,
        reranker=RAGRerankerConfig(type="llm", top_n=2),
    )
    manifest.models = _rerank_models(models)
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" in result


# ===========================================================================
# 5. Namespace isolation
# ===========================================================================


async def test_rag_namespace_isolates_documents_in_shared_collection(
    client, models, embeds, embed_seed_base_url, rag_redis_urls
):
    """
    Two RAGSources pointing at the *same* Redis collection but different
    `namespace` values must not see each other's documents - proves the
    @namespace:{...} TAG filter in _RedisBackend.search actually isolates.
    """
    collection = f"rag-e2e-shared-{uuid.uuid4().hex[:12]}"
    ns_a, ns_b = f"ns-a-{uuid.uuid4().hex[:8]}", f"ns-b-{uuid.uuid4().hex[:8]}"
    embed_cfg = embeds[0]

    await seed_redis_rag_source(
        redis_url=rag_redis_urls[0], collection=collection, namespace=ns_a,
        documents=["The secret launch code for Project Aurora is 'blue-owl'."],
        embed_base_url=embed_seed_base_url, embed_model=embed_cfg.model,
        embed_api_key=embed_cfg.api_key,
    )
    await seed_redis_rag_source(
        redis_url=rag_redis_urls[0], collection=collection, namespace=ns_b,
        documents=["The weather in Reykjavik in July averages 12°C."],
        embed_base_url=embed_seed_base_url, embed_model=embed_cfg.model,
        embed_api_key=embed_cfg.api_key,
    )

    try:
        source_b = RAGSource(
            name="docs", backend="redis", collection=collection,
            connection_url=rag_redis_urls[1], namespace=ns_b, top_k=4,
        )
        manifest = _rag_manifest("RagNamespaceIsolationAgent", models, embeds, source_b)
        thread = await client.aelite.threads.create_thread(manifest=manifest)

        session = await client.aelite.connect(thread.id)
        result = await session.chat("What is the secret launch code for Project Aurora?")

        assert isinstance(result, str)
        assert "blue-owl" not in result.lower()
    finally:
        await drop_redis_rag_source(rag_redis_urls[0], collection)


# ===========================================================================
# 6. Master switch
# ===========================================================================


async def test_rag_disabled_never_injects(client, models, embeds, rag_source):
    """rag.enabled=False: {rag_<name>} must not appear resolved in the
    prompt - the model has no way to know about the seeded documents."""
    manifest = _rag_manifest("RagDisabledAgent", models, embeds, rag_source)
    manifest.rag.enabled = False
    thread = await client.aelite.threads.create_thread(manifest=manifest)

    session = await client.aelite.connect(thread.id)
    result = await session.chat("How long does the Zenith X200's battery last?")

    assert isinstance(result, str)
    assert "40" not in result


# ===========================================================================
# 7. Config validation (pure Pydantic, no server round-trip needed)
# ===========================================================================


class TestRagRerankerValidation:
    def _base_kwargs(self, **overrides):
        kwargs = dict(
            id="RagValidationAgent",
            system_prompt="You are helpful.",
            models=[
                ModelConfig(base_url="https://example.com", model="m", api_key="k")
            ],
            embeds=[
                EmbedConfig(base_url="https://example.com", model="e", api_key="k")
            ],
        )
        kwargs.update(overrides)
        return kwargs

    def test_cross_encoder_without_rerankers_raises(self):
        with pytest.raises(ValueError, match="rerankers.*use_for_rag"):
            AgentManifest(
                **self._base_kwargs(
                    rag=RAGConfig(
                        enabled=True,
                        sources=[
                            RAGSource(name="d", backend="redis", collection="c", connection_url="redis://localhost:6379")
                        ],
                        reranker=RAGRerankerConfig(type="cross_encoder"),
                    ),
                )
            )

    def test_cross_encoder_with_rerankers_is_valid(self):
        manifest = AgentManifest(
            **self._base_kwargs(
                rerankers=[
                    RerankerConfig(
                        base_url="https://example.com", model="r", api_key="k"
                    )
                ],
                rag=RAGConfig(
                    enabled=True,
                    sources=[RAGSource(name="d", backend="redis", collection="c", connection_url="redis://localhost:6379")],
                    reranker=RAGRerankerConfig(type="cross_encoder"),
                ),
            )
        )
        assert manifest.rag.reranker.type == "cross_encoder"

    def test_llm_reranker_without_use_for_rerank_model_raises(self):
        with pytest.raises(ValueError, match="use_for_rerank"):
            AgentManifest(
                **self._base_kwargs(
                    rag=RAGConfig(
                        enabled=True,
                        sources=[RAGSource(name="d", backend="redis", collection="c", connection_url="redis://localhost:6379")],
                        reranker=RAGRerankerConfig(type="llm"),
                    ),
                )
            )

    def test_llm_reranker_with_use_for_rerank_model_is_valid(self):
        kwargs = self._base_kwargs(
            rag=RAGConfig(
                enabled=True,
                sources=[RAGSource(name="d", backend="redis", collection="c", connection_url="redis://localhost:6379")],
                reranker=RAGRerankerConfig(type="llm"),
            ),
        )
        kwargs["models"][0].use_for_rerank = True
        manifest = AgentManifest(**kwargs)
        assert manifest.rag.reranker.type == "llm"

    def test_none_and_rrf_require_no_extra_config(self):
        for rtype in ("none", "rrf"):
            manifest = AgentManifest(
                **self._base_kwargs(
                    rag=RAGConfig(
                        enabled=True,
                        sources=[RAGSource(name="d", backend="redis", collection="c", connection_url="redis://localhost:6379")],
                        reranker=RAGRerankerConfig(type=rtype),
                    ),
                )
            )
            assert manifest.rag.reranker.type == rtype

    def test_rag_disabled_skips_reranker_validation_entirely(self):
        """A cross_encoder/llm reranker misconfiguration with rag.enabled=False
        must not raise - the reranker is never actually invoked."""
        manifest = AgentManifest(
            **self._base_kwargs(
                rag=RAGConfig(
                    enabled=False,
                    reranker=RAGRerankerConfig(type="cross_encoder"),
                ),
            )
        )
        assert manifest.rag.enabled is False

    def test_reranker_model_field_removed(self):
        """RAGRerankerConfig no longer has a `model` field - the model now
        lives on RerankerConfig entries in manifest.rerankers."""
        cfg = RAGRerankerConfig(type="cross_encoder", top_n=5)
        assert not hasattr(cfg, "model")
