"""
rag_redis.py - test-side seeding helper for RAG's backend='redis' source.

_RedisBackend (app/services/agent_factory/rag_service.py) assumes the
RediSearch index already exists - unlike A-MEM's _RedisNoteStore, it has
no _ensure_index step. So e2e RAG tests own index creation/teardown
themselves, exactly as an operator would provision it in production.

Schema mirrors what _RedisBackend.search actually queries:
    - "content" TEXT
    - "namespace" TAG            (queried as @namespace:{value})
    - any metadata_filter keys   (queried as @{key}:{value}, so must be TAG)
    - "embedding" VECTOR FLAT FLOAT32, dim = embedding provider's dimension
"""

from __future__ import annotations

import uuid

import httpx
import numpy as np
from redis.asyncio import Redis
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType


async def embed_texts(
    base_url: str, model: str, api_key: str, texts: list[str]
) -> list[list[float]]:
    """Calls the same OpenAI-compatible /v1/embeddings endpoint the server's
    ResilientEmbedModel calls, so vectors are guaranteed compatible with
    whatever `embeds` fixture the test is using."""
    async with httpx.AsyncClient(base_url=base_url) as client:
        resp = await client.post(
            "/embeddings",
            json={"input": texts, "model": model},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]


async def seed_redis_rag_source(
    redis_url: str,
    collection: str,
    namespace: str,
    documents: list[str],
    embed_base_url: str,
    embed_model: str,
    embed_api_key: str,
    extra_tag_fields: list[str] | None = None,
) -> None:
    """Creates a RediSearch index named `{collection}` (RAGSource.collection
    IS the index name for backend='redis'/'valkey' - see RAGSource.collection
    docstring) and stores `documents`, each embedded and tagged with `namespace`."""
    client = Redis.from_url(redis_url)
    idx_name = collection

    vectors = await embed_texts(embed_base_url, embed_model, embed_api_key, documents)
    dim = len(vectors[0])

    try:
        await client.ft(idx_name).info()
    except Exception:
        schema = [
            TextField("content"),
            TagField("namespace"),
            *[TagField(f) for f in (extra_tag_fields or [])],
            VectorField(
                "embedding", "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        definition = IndexDefinition(
            prefix=[f"{collection}:doc:"], index_type=IndexType.HASH,
        )
        await client.ft(idx_name).create_index(schema, definition=definition)

    for i, (text, vec) in enumerate(zip(documents, vectors)):
        key = f"{collection}:doc:{uuid.uuid4()}"
        mapping = {
            "content": text,
            "namespace": namespace,
            "embedding": np.array(vec, dtype=np.float32).tobytes(),
        }
        await client.hset(key, mapping=mapping)

    await client.aclose()


async def drop_redis_rag_source(redis_url: str, collection: str) -> None:
    client = Redis.from_url(redis_url)
    idx_name = collection
    try:
        await client.ft(idx_name).dropindex(delete_documents=True)
    except Exception:
        pass  # already gone / never created - nothing to clean up
    await client.aclose()