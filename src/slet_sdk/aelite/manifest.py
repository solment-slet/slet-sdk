"""
manifest.py — Declarative agent configuration schema.

This module defines the complete Pydantic model hierarchy used to describe
an agent and all its runtime behaviour: LLM parameters, persistent memory,
RAG retrieval, node-level caching, A-MEM knowledge graph, dynamic tool
retrieval, tool inheritance, triggers, and peripheral components.

A manifest is submitted once via ``POST /agent/deploy/{session_id}``,
serialised to Redis, and used by ``AgentFactory`` to compile a LangGraph
graph that is cached in memory per replica. Sub-agents are fully recursive —
each sub-agent is itself an ``AgentManifest`` and is deployed as an
independent entity with its own ``session_id`` and WebSocket endpoint.

Credential handling
-------------------
Fields that may contain sensitive connection strings (``connection_url``)
are marked ``exclude=True`` so they are never written to Redis or returned
in API responses. Prefer ``connection_url_env`` (the name of a server-side
environment variable) over inline credentials.

System-prompt variables
-----------------------
The following placeholders are expanded before every LLM call:

- ``{time}`` — current server time (always available)
- ``{summary}`` — rolling conversation summary
                          (requires ``memory.enabled = True`` and
                          ``memory.summarization`` set to ``'async'``
                          or ``'blocking'``)
- ``{amem}`` — A-MEM knowledge-graph notes
                          (requires ``memory.agentic.enabled = True``;
                          variable name overridable via ``inject_variable``)
- ``{rag_<name>}`` — RAG chunks for the source whose ``name`` field
                          equals ``<name>`` (requires ``rag.enabled = True``
                          and ``inject_as = 'system_variable'``)

Unknown placeholders produce an inline error note instead of crashing.
"""

from __future__ import annotations

from typing import Any, Literal, Annotated, Callable
from enum import StrEnum, Enum

from pydantic import BaseModel, Field, model_validator, field_validator

from slet_sdk.aelite.tools import ToolBelt


# ===========================================================================
# Permissions
# ===========================================================================


class PermissionLevel(str, Enum):
    owner_only = "owner_only"
    whitelist = "whitelist"
    public = "public"


class PermissionLevelRestricted(str, Enum):
    owner_only = "owner_only"
    whitelist = "whitelist"


class AgentPermissions(BaseModel):
    """
    Access control policy for a root agent.

    Each field defines who is allowed to perform a specific action:

    - ``owner_only``: only the user who deployed the agent.
    - ``whitelist``: only users explicitly listed in ``allowed_users``.
    - ``public``: any authenticated user.

    ``deploy`` and ``delete`` do not support ``public`` — granting arbitrary
    users the ability to overwrite or destroy an agent is intentionally
    disallowed.
    """

    view_manifest: PermissionLevel = Field(
        default=PermissionLevel.owner_only,
        description=(
            "Who can retrieve this agent's manifest via ``GET /agent/{agent}``. "
            "Defaults to ``owner_only`` — manifests are private unless explicitly "
            "opened. Set to ``public`` to make the agent's configuration visible "
            "to any authenticated user, or ``whitelist`` to share with a specific group."
        ),
    )
    create_thread: PermissionLevel = Field(
        default=PermissionLevel.owner_only,
        description=(
            "Who can start a new conversation thread backed by this agent "
            "via ``POST /threads/``. Defaults to ``owner_only``, making the agent "
            "personal by default. Set to ``public`` to allow any authenticated user "
            "to use the agent, or ``whitelist`` for invite-only access."
        ),
    )
    deploy: PermissionLevelRestricted = Field(
        default=PermissionLevelRestricted.owner_only,
        description=(
            "Who can push a new version of this agent via ``PUT /agent/{agent}``. "
            "Defaults to ``owner_only``. Set to ``whitelist`` to allow a team "
            "of developers to deploy updates. ``public`` is not permitted — "
            "unrestricted write access to agent logic is a security risk."
        ),
    )
    delete: PermissionLevelRestricted = Field(
        default=PermissionLevelRestricted.owner_only,
        description=(
            "Who can permanently delete this agent. "
            "Defaults to ``owner_only``. ``public`` is not permitted."
        ),
    )
    allowed_users: list[int] = Field(
        default_factory=list,
        description=(
            "Explicit list of user IDs granted access when any permission field "
            "is set to ``whitelist``. A single list applies across all whitelist "
            "permissions — if granular per-action whitelists are needed, "
            "deploy separate agents with different configurations."
        ),
    )


# ===========================================================================
# Concurrency
# ===========================================================================


class SequentialTimeoutPolicy(StrEnum):
    ignore = "ignore"                # processing_timeout is ignored
    release_lock = "release_lock"    # the lock is released and the following request is executed
    cancel = "cancel"                # request is canceled


class SequentialExecutionConfig(BaseModel):
    """
    Configuration specific to sequential execution mode.

    Controls how requests are serialized, how long they may wait or run,
    and what action is taken when the processing timeout expires.
    """

    processing_timeout: int = Field(
        default=120,
        ge=1,
        description=(
            "Maximum execution time, in seconds, before ``timeout_policy`` is "
            "applied. Increase for slow client tools or long LLM calls. Must "
            "be greater than any tool's ``timeout``."
        )
    )
    queue_timeout: int = Field(
        default=60,
        ge=1,
        description=(
            "Maximum seconds a new request waits to acquire the processing "
            "lock (sequential mode only). After this, the client receives "
            "``503 Agent is busy``."
        ),
    )
    timeout_policy: SequentialTimeoutPolicy = Field(
        default=SequentialTimeoutPolicy.release_lock,
        description="Defines the action to take when ``processing_timeout`` expires."
    )


class ConcurrencyConfig(BaseModel):
    """
    Controls how requests are executed for this agent.

    Defines the execution mode and any mode-specific configuration, such as
    sequential locking and timeout behavior.
    """

    mode: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description=(
            "Message processing strategy:\n"
            " - ``'sequential'``: Redis lock serialises all requests for this "
            "thread across replicas.\n"
            " - ``'parallel'``: no router-level lock. Atomicity is enforced "
            "by the checkpointer's short write-lock."
        ),
    )
    sequential: SequentialExecutionConfig = Field(
        default_factory=SequentialExecutionConfig,
        description=(
            "Settings that apply only when ``mode`` is ``'sequential'``."
        )
    )


# ===========================================================================
# MCP
# ===========================================================================


class MCPResourceMode(StrEnum):
    """
    Defines how MCP server resources are exposed to the agent.

    tool
        Each resource is registered as a pseudo-tool named
        ``<server_id>_resource_<name>``. The LLM decides when to request
        the resource. Suitable for optional or expensive resources that
        are not required on every request.

    tool_with_retriever
        Same as ``tool``, but additionally registers a retriever tool
        ``<server_id>_find_resource``. The retriever accepts a text query
        and returns matching resources. Useful when a server exposes a
        large number of resources and the LLM needs assistance discovering
        the relevant ones.

    system_prompt
        Resource contents are injected into the system prompt via
        placeholders of the form
        ``{mcp_res_<server_id>_<resource_name>}``.
        The LLM always sees the resource content. Intended for
        reasonably sized textual resources.

    message
        Resource contents are injected into the current user message
        through the same placeholders. Useful for turn-specific context.
    """

    tool = "tool"
    tool_with_retriever = "tool_with_retriever"
    system_prompt = "system_prompt"
    message = "message"


class MCPTransportSSE(BaseModel):
    type: Literal["sse"] = "sse"
    url: str | None = Field(
        default=None,
        description=(
            "Base URL of the remote MCP server. Required when using "
            "the SSE transport."
        ),
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional HTTP headers sent with requests "
            "(for example authentication headers)."
        ),
    )

    @model_validator(mode="after")
    def validate_transport_fields(self):
        if not self.url:
            raise ValueError("url is required")
        return self


class MCPTransportStdio(BaseModel):
    type: Literal["stdio"] = "stdio"
    command: str | None = Field(
        default=None,
        description=(
            "Command used to launch the local stdio process on the client. "
            "Stored only for documentation and manifest validation. "
            "The server identifies the MCP server exclusively by ``id``."
        ),
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments passed to ``command``.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Environment variables supplied to the process. "
            "Included for documentation purposes only."
        ),
    )

    @model_validator(mode="after")
    def validate_transport_fields(self):
        if not self.command:
            raise ValueError("command is required")
        return self


class MCPServerConfig(BaseModel):
    """
    Configuration of a single MCP server attached to the agent.

    Two transport modes are supported:

    **SSE** (``transport='sse'``)
        The backend connects directly to the remote MCP server over HTTP.
        Requires ``url``.

    **stdio** (``transport='stdio'``)
        The backend sends an ``mcp_call`` event to the client over
        WebSocket. The client launches a local stdio process and returns
        the result via ``mcp_result``. The client maintains its own
        mapping of ``server_id → command``. The backend transmits only
        ``server_id``, preventing arbitrary command execution on the
        client side.

        ``command`` is required for documentation and manifest validation
        but is never transmitted to the client.

    Tools, resources, and prompts provided by the server are integrated
    into the agent alongside declarative ``tools`` defined in the manifest
    and participate in ``inherit_tools_from`` resolution.
    """

    id: str = Field(
        description=(
            "Unique server identifier within the agent. Used as a prefix "
            "for generated tool names (``<id>_<tool_name>``) and resource "
            "placeholders (``{mcp_res_<id>_<resource_name>}``) to avoid "
            "name collisions when multiple MCP servers are attached."
        ),
    )
    transport: Annotated[
        MCPTransportSSE | MCPTransportStdio,
        Field(discriminator="type")
    ] = Field(
        default_factory=MCPTransportSSE,
        description=(
            "Transport configuration. SSE connects directly over HTTP, "
            "while stdio is proxied through the client."
        ),
    )

    # Content to load
    load_tools: bool = Field(
        default=True,
        description=(
            "Load and register tools exposed by the server "
            "(``tools/list``)."
        ),
    )
    load_resources: bool = Field(
        default=False,
        description=(
            "Load resources exposed by the server "
            "(``resources/list``)."
        ),
    )
    resource_mode: MCPResourceMode = Field(
        default=MCPResourceMode.tool,
        description=(
            "Controls how resources are exposed to the agent. "
            "Ignored when ``load_resources=False``. "
            "See ``MCPResourceMode`` for details."
        ),
    )
    load_prompts: bool = Field(
        default=False,
        description=(
            "Load prompts exposed by the server (``prompts/list``). "
            "Loaded prompts are available through placeholders of the form "
            "``{mcp_prompt_<id>_<prompt_name>}`` in the system prompt "
            "or any message."
        ),
    )

    timeout: int = Field(
        default=30,
        ge=1,
        description="Timeout for a single MCP request in seconds.",
    )


# ===========================================================================
# Broadcast
# ===========================================================================


class BroadcastMode(StrEnum):
    """
    Determines how the server collects responses from connected WebSocket
    clients when a client tool is invoked in broadcast mode.

    Modes
    -----
    disabled
        Broadcast is off. The tool call is delivered only to the connection
        that triggered the current agent run (via ``current_websocket``).
    first
        The server resolves as soon as **any single** client responds.
        Remaining responses are silently discarded. Lowest latency option.
    collect
        The server waits for the first response, then holds the result open
        for an additional ``collect_window`` seconds to gather stragglers
        before resolving. Good balance between completeness and latency.
    threshold
        The server waits until at least ``threshold_percent`` % of currently
        connected clients have responded, then waits an additional
        ``collect_window`` seconds before resolving.
        Example: 3 clients connected, ``threshold_percent=67`` → waits for 2.
    all
        The server waits for **every** connected client to respond or for
        ``timeout`` to expire, whichever comes first. Guarantees full
        participation when all clients are reliable; use with a generous
        ``timeout``.
    """

    disabled = "disabled"
    first = "first"
    collect = "collect"
    threshold = "threshold"
    all = "all"


class BroadcastConfig(BaseModel):
    """
    Configuration for broadcast client tool calls.

    When a client tool is invoked in broadcast mode, the server publishes a
    ``client_tool_call`` event to **all** WebSocket connections on the current
    ``session_id`` (across all pods via Redis pub/sub) and then aggregates the
    responses according to ``mode`` before returning a single string to the LLM.

    The aggregated result always follows this format::

        [BROADCAST] Received {n}/{total} response(s):
          [1] <result>
          [2] <result>
          (k client(s) did not respond within {timeout}s) # if any timed out

    Notes
    -----
    - ``expected`` is snapshot at call time: the number of active WebSocket
      connections to ``session_id`` when the tool is invoked. Clients that
      connect or disconnect mid-call are not accounted for.
    - ``collect_window`` is only meaningful for ``collect`` and ``threshold``
      modes; it is ignored for ``first``, ``all``, and ``disabled``.
    - ``threshold_percent`` is only meaningful for ``threshold`` mode.
    - The overall deadline is always ``timeout`` seconds regardless of mode.
      ``collect_window`` cannot extend beyond it.
    """

    mode: BroadcastMode = Field(
        default=BroadcastMode.all,
        description=(
            "Response collection strategy. See ``BroadcastMode`` for full "
            "semantics of each option."
        ),
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Hard deadline in seconds for the entire broadcast round-trip: "
            "from sending ``client_tool_call`` to returning the aggregated "
            "result to the LLM. Clients that have not responded by this time "
            "are counted as non-respondents in the summary."
        ),
    )
    collect_window: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Extra seconds to wait for additional responses after the primary "
            "condition is met (first response for ``collect``, threshold reached "
            "for ``threshold``). Capped by the remaining ``timeout`` budget. "
            "Ignored in ``first``, ``all``, and ``disabled`` modes."
        ),
    )
    threshold_percent: float = Field(
        default=50.0,
        description=(
            "Minimum percentage of connected clients that must respond before "
            "the server moves to the ``collect_window`` phase. Only used in "
            "``threshold`` mode. Must be in the range (0, 100]. "
            "Example: 4 clients connected, ``threshold_percent=75`` → waits "
            "for at least 3 responses."
        ),
    )
    include_client_id: bool = Field(
        default=False,
        description=(
            "When ``True``, each response line in the aggregated result is "
            "prefixed with the client's identifier as reported in the "
            "``client_tool_result`` payload. Useful when the LLM needs to "
            "attribute responses to specific participants."
        ),
    )

    @field_validator("threshold_percent")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0 < v <= 100:
            raise ValueError("threshold_percent must be in the range (0, 100]")
        return v


# ===========================================================================
# Model
# ===========================================================================


class ProviderKind(StrEnum):
    OPENAI_COMPATIBLE = "openai_compatible"
    ANTHROPIC = "anthropic"


class ModelConfig(BaseModel):
    """
    Per-agent LLM sampling parameters.
    """
    base_url: str | None = Field(
        default=None,
        min_length=8,
        max_length=500,
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    api_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=512,
    )
    kind: ProviderKind = Field(
        default=ProviderKind.OPENAI_COMPATIBLE,
        description=(
            "Protocol/SDK to use for this custom provider. Defaults "
            "to the OpenAI-compatible chat completions API."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. Lower values produce more deterministic "
            "output; higher values increase diversity. Typical range: "
            "0.0 (greedy) to 1.0 (balanced)."
        ),
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Nucleus sampling threshold. Tokens are sampled from the smallest "
            "set whose cumulative probability exceeds ``top_p``. Avoid setting "
            "both ``temperature`` and ``top_p`` simultaneously."
        ),
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Hard cap on the number of tokens the model may generate per "
            "response. Prevents runaway generation on open-ended prompts."
        ),
    )
    is_reasoning_model: bool = Field(
        default=False,
        description=(
            "Mark this custom model as an OpenAI o1/o3-style reasoning model. "
            "When true, temperature/top_p are omitted from requests, since "
            "these models reject them."
        ),
    )
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Reasoning effort for OpenAI-style reasoning models (o1/o3 and "
            "compatible proxies). Higher effort trades latency and cost for "
            "better reasoning quality. Ignored by non-reasoning models. "
            "Only applicable to ProviderKind.OPENAI_COMPATIBLE — Anthropic "
            "has no equivalent parameter (use thinking_budget_tokens instead)."
        ),
    )
    thinking_enabled: bool = Field(
        default=False,
        description=(
            "Enable extended thinking / reasoning mode (Anthropic extended "
            "thinking, DeepSeek-R1, and similar). When enabled, the model "
            "emits intermediate reasoning steps before its final answer. "
            "Note: for Anthropic models this forces temperature=1 and "
            "disables top_p, per the provider's API constraints. For "
            "OPENAI_COMPATIBLE providers, this requires is_reasoning_model=True "
            "or reasoning_effort to be set, otherwise it has no effect."
        ),
    )
    thinking_budget_tokens: int | None = Field(
        default=None,
        ge=1024,
        description=(
            "Token budget allocated to the model's internal reasoning pass. "
            "Only applies when thinking_enabled=True. Primarily relevant to "
            "Anthropic extended thinking; ignored by providers that don't "
            "support a configurable reasoning budget."
        ),
    )

    @model_validator(mode="after")
    def _validate_custom_provider_completeness(self):
        """
        base_url/model/api_key must be provided all together (a fully
        specified custom provider) or all omitted (fall back to the
        server's global provider pool). Partial specification is always
        a configuration mistake.
        """
        provided = [self.base_url is not None, self.model is not None, self.api_key is not None]
        if any(provided) and not all(provided):
            raise ValueError(
                "base_url, model and api_key must all be provided together "
                "for a custom provider entry, or all omitted to fall back "
                "to the server's default provider pool."
            )
        return self

    @model_validator(mode="after")
    def _validate_thinking_requires_reasoning_mode(self):
        """
        For ProviderKind.OPENAI_COMPATIBLE, thinking_enabled only has an
        effect inside the `if is_reasoning_model:` branch in _build_llm
        (app/services/model_manager.py) — that's where extra_body with
        reasoning_format is attached. If is_reasoning_model=False and
        reasoning_effort is unset, that whole branch is skipped and
        thinking_enabled is silently a no-op: the request never asks the
        provider for reasoning content, and no thinking is ever returned,
        with no error surfaced anywhere.

        Does not apply to ProviderKind.ANTHROPIC, where thinking_enabled
        drives its own independent `thinking` API parameter.
        """
        if (
            self.kind == ProviderKind.OPENAI_COMPATIBLE
            and self.thinking_enabled
            and not self.is_reasoning_model
            and not self.reasoning_effort
        ):
            raise ValueError(
                "thinking_enabled=True has no effect for an OPENAI_COMPATIBLE "
                "provider unless is_reasoning_model=True or reasoning_effort "
                "is set — otherwise the reasoning_format request parameter is "
                "never sent and no reasoning/thinking content will be "
                "returned. Set is_reasoning_model=True (and/or reasoning_effort) "
                "if this model is a reasoning model, or set thinking_enabled=False."
            )
        return self

    @model_validator(mode="after")
    def _validate_reasoning_effort_provider(self):
        """
        reasoning_effort is an OpenAI-style reasoning API concept (o1/o3/
        gpt-oss and OpenAI-compatible proxies). Anthropic's extended
        thinking is controlled via thinking_enabled + thinking_budget_tokens
        instead — _build_llm never reads reasoning_effort for
        ProviderKind.ANTHROPIC, so setting it there is silently ignored.
        """
        if self.kind == ProviderKind.ANTHROPIC and self.reasoning_effort:
            raise ValueError(
                "reasoning_effort has no effect for kind=ANTHROPIC — Anthropic "
                "extended thinking is controlled via thinking_enabled and "
                "thinking_budget_tokens instead. Remove reasoning_effort or "
                "use thinking_budget_tokens to control reasoning depth."
            )
        return self


# ===========================================================================
# Model Rotation
# ===========================================================================


class ModelRotationScope(StrEnum):
    thread = "thread"
    """
    Default. Fallback/rotation progress within ``AgentManifest.model`` is
    shared across every WebSocket connection attached to this thread on a
    given pod — mirrors the pre-existing behaviour where a single
    ResilientChatModel instance served the whole thread. Appropriate when
    ``model`` describes infrastructure the operator controls (e.g. the
    server's own provider pool, or a custom chain shared by a team), rather
    than per-user credentials.
    """
    connection = "connection"
    """
    Each WebSocket connection tracks its own fallback/rotation progress
    independently, even for the same thread_id. Required whenever different
    connections to the same thread may belong to different people with
    different credentials or preferences — otherwise one connection's dead
    custom provider would silently degrade another connection's requests.
    """


class ModelRotationPolicy(BaseModel):
    """
    Controls how ``AgentManifest.model`` fallback state is shared (or not)
    across the WebSocket connections attached to a thread, and whether a
    connecting client may supply its own model chain instead of using the
    manifest's.
    """

    scope: ModelRotationScope = Field(
        default=ModelRotationScope.thread,
        description=(
            "Whether model fallback/rotation progress is shared across all "
            "connections to this thread, or isolated per WebSocket "
            "connection. Forced to per-connection isolation automatically "
            "for any connection that has an active client-supplied override "
            "(see allow_client_override) regardless of this setting — a "
            "client-supplied chain, and any credentials embedded in it, "
            "must never be shared with other connections on the same thread."
        ),
    )
    allow_client_override: bool = Field(
        default=True,
        description=(
            "When True, a connecting client may send a 'set_models' event "
            "over the WebSocket (before or between 'message'/'trigger' "
            "events) carrying its own list[ModelConfig], which entirely "
            "replaces AgentManifest.model for that connection's requests. "
            "Rotation for that connection is then always isolated "
            "per-connection, regardless of the configured scope value. "
            "When this is True and AgentManifest.model is empty, every "
            "connection MUST send 'set_models' before sending its first "
            "'message'/'trigger', since there is no server-side default "
            "chain to fall back to — see AgentManifest.model's docstring."
        ),
    )


# ===========================================================================
# RAG — Retrieval-Augmented Generation
# ===========================================================================


class RAGSource(BaseModel):
    """
    A single vector-store data source used for retrieval.

    An agent may declare multiple sources; retrieval requests are issued to
    all of them in parallel and the results are merged by the configured
    ``RerankerConfig`` before being injected into the context.

    Connection resolution order:
    1. ``connection_url`` (inline, excluded from serialisation)
    2. ``connection_url_env`` → ``os.environ[connection_url_env]``
    3. Environment variable ``RAG_{BACKEND}_URL`` (e.g. ``RAG_QDRANT_URL``)
    4. For ``backend = 'redis'`` only: fall back to the server-managed Redis
       client (``common.redis``).
    """

    name: str = Field(
        description=(
            "Logical source identifier. Used as the suffix in the system-prompt "
            "variable ``{rag_<name>}`` when ``inject_as = 'system_variable'``, "
            "and as the tool name ``<name>_search`` when ``inject_as = 'tool'``. "
            "Must be unique within the agent's ``rag.sources`` list."
        ),
    )
    backend: Literal["pgvector", "qdrant", "chroma", "redis"] = Field(
        description=(
            "Vector-store backend to query. Each backend requires its own "
            "Python package to be installed on the server:\n"
            " - ``'pgvector'``: PostgreSQL + pgvector extension (``asyncpg``).\n"
            " - ``'qdrant'``: Qdrant vector database (``qdrant-client``).\n"
            " - ``'chroma'``: ChromaDB (``chromadb``).\n"
            " - ``'redis'``: Redis Stack with vector-similarity search (``redis``)."
        ),
    )
    collection: str = Field(
        description=(
            "Collection, index, or table name inside the backend. Interpretation "
            "is backend-specific: a Qdrant collection name, a pgvector table name, "
            "a Chroma collection, or a Redis index name."
        ),
    )
    connection_url: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "Full connection URL for the backend (e.g. ``postgresql+asyncpg://...``, "
            "``http://qdrant-host:6333``). Excluded from serialisation — prefer "
            "``connection_url_env`` to avoid storing credentials in Redis."
        ),
    )
    connection_url_env: str | None = Field(
        default=None,
        description=(
            "Name of a server-side environment variable that holds the connection "
            "URL. Resolved at retrieval time via ``os.environ[connection_url_env]``. "
            "Safer than ``connection_url`` because the value never leaves the "
            "server process."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Maximum number of chunks to retrieve from this source per query. "
            "The reranker may reduce the final count further via ``top_n``."
        ),
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine-similarity score. Chunks below this threshold are "
            "discarded before reranking. ``None`` disables score filtering — "
            "all ``top_k`` results are kept regardless of quality."
        ),
    )
    namespace: str | None = Field(
        default=None,
        description=(
            "Optional namespace or tenant key applied as a metadata filter on "
            "every query. Useful for multi-tenant deployments where a single "
            "collection stores documents for multiple customers."
        ),
    )
    metadata_filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Static metadata filter applied to every query against this source. "
            "Format is backend-specific:\n"
            " - Qdrant: ``{'must': [{'key': 'lang', 'match': {'value': 'en'}}]}``\n"
            " - pgvector: key-value pairs passed as WHERE-clause fragments.\n"
            " - Chroma: ``{'lang': {'$eq': 'en'}}``"
        ),
    )
    embed_query_template: str | None = Field(
        default=None,
        description=(
            "Template applied to the raw query text before embedding. "
            "``{query}`` is replaced with the actual query. Useful for query "
            "expansion (e.g. ``'Represent this question for retrieval: {query}'``) "
            "or HyDE (Hypothetical Document Embeddings)."
        ),
    )
    inject_as: Literal["system_variable", "tool"] = Field(
        default="system_variable",
        description=(
            "How retrieved chunks are exposed to the LLM:\n"
            " - ``'system_variable'``: chunks are inserted into the system "
            "prompt via ``{rag_<name>}``. The LLM always sees the retrieved "
            "context.\n"
            " - ``'tool'``: a server-side tool ``<name>_search`` is registered "
            "dynamically. The LLM decides when to call it. Saves tokens when "
            "retrieval is only occasionally needed."
        ),
    )
    tool_description: str | None = Field(
        default=None,
        description=(
            "Description shown to the LLM when ``inject_as = 'tool'``. Should "
            "explain when to call this tool and what kinds of questions it can "
            "answer. If ``None``, a generic description is generated from the "
            "source ``name``."
        ),
    )


class RerankerConfig(BaseModel):
    """
    Post-retrieval reranking applied after all ``RAGSource`` results are
    collected. Results from multiple sources are pooled before reranking
    so the final ``top_n`` chunks represent the best matches overall.
    """

    type: Literal["cross_encoder", "llm", "rrf", "none"] = Field(
        default="none",
        description=(
            "Reranking algorithm:\n"
            " - ``'none'``: concatenate in source order, truncate to ``top_n``.\n"
            " - ``'rrf'``: Reciprocal Rank Fusion — fast, no neural model.\n"
            " - ``'cross_encoder'``: local cross-encoder model (``sentence-transformers``).\n"
            " - ``'llm'``: primary LLM acts as a relevance judge. Most accurate "
            "but expensive."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Model identifier for ``'cross_encoder'`` and ``'llm'`` rerankers. "
            "For cross-encoders, a HuggingFace model name "
            "(e.g. ``'cross-encoder/ms-marco-MiniLM-L-6-v2'``). "
            "Ignored for ``'rrf'`` and ``'none'``."
        ),
    )
    top_n: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of chunks to retain after reranking. The final context "
            "contains at most ``top_n`` chunks regardless of how many sources "
            "contributed."
        ),
    )


class RAGConfig(BaseModel):
    """
    Retrieval-Augmented Generation configuration.

    When enabled, ``RAGService`` issues parallel queries to all declared
    sources, merges results through the reranker, and makes chunks available
    to the LLM either via system-prompt injection or dynamic search tools.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When ``False``, no retrieval is performed and "
            "``{rag_*}`` placeholders expand to empty strings."
        ),
    )
    sources: list[RAGSource] = Field(
        default_factory=list,
        description=(
            "Ordered list of vector-store sources. Queries are issued to all "
            "sources in parallel. At least one source is required when "
            "``enabled = True``."
        ),
    )
    reranker: RerankerConfig = Field(
        default_factory=RerankerConfig,
        description=(
            "Reranking strategy applied after all source results are collected. "
            "Defaults to ``type = 'none'`` (concatenate and truncate)."
        ),
    )
    query_mode: Literal["last_message", "summary", "custom_tool"] = Field(
        default="last_message",
        description=(
            "What text is used as the retrieval query:\n"
            " - ``'last_message'``: content of the most recent HumanMessage.\n"
            " - ``'summary'``: current rolling summary from ``SummaryStore``. "
            "Useful when relevant context spans many turns.\n"
            " - ``'custom_tool'``: retrieval is not triggered automatically; "
            "the agent calls ``<name>_search`` tools explicitly. Requires all "
            "sources to use ``inject_as = 'tool'``."
        ),
    )
    max_tokens_per_source: int | None = Field(
        default=2000,
        ge=100,
        description=(
            "Hard token limit for the context injected per source. Chunks are "
            "trimmed (not dropped) to fit. ``None`` disables trimming. Does "
            "not apply to sources with ``inject_as = 'tool'``."
        ),
    )

    @model_validator(mode="after")
    def validate_sources_present(self):
        if self.enabled and not self.sources:
            raise ValueError("rag.enabled=True requires at least one source.")
        if self.query_mode == "custom_tool":
            if any(s.inject_as != "tool" for s in self.sources):
                raise ValueError(
                    "query_mode='custom_tool' requires all sources inject_as='tool'."
                )
        return self


# ===========================================================================
# Cache
# ===========================================================================


class NodeCachePolicy(BaseModel):
    """
    LangGraph ``CachePolicy`` settings for a single graph node.

    Controls whether (and for how long) a node's output is cached based on
    its input state. A cache hit skips the node function entirely, returning
    the cached output directly.
    """

    ttl: int | None = Field(
        default=300,
        ge=1,
        description=(
            "Cache entry time-to-live in seconds. After this period the entry "
            "is evicted and the node re-executes on the next request. ``None`` "
            "means entries never expire — use only for truly static computations."
        ),
    )
    key_fields: list[str] | None = Field(
        default=None,
        description=(
            "State field paths included in the cache key. Each entry is a "
            "dot-notation accessor evaluated against the node's input state "
            "(e.g. ``'messages[-1].content'``). ``None`` uses the full "
            "serialised state hash — safe but may have a lower hit rate when "
            "irrelevant fields change between requests."
        ),
    )


class CacheConfig(BaseModel):
    """
    Multi-layer caching configuration.

    Two independent layers are configured here:

    1. **Node cache**: LangGraph-level ``CachePolicy`` on individual graph
       nodes (``rag_retrieval``, ``amem_retrieval``). The shared cache
       backend is selected via ``backend``.

    2. **Prompt cache**: provider-side prefix caching (Anthropic
       ``cache_control``). The static portion of the system prompt is
       billed at ~10% of the normal input-token rate on cache hits.
    """

    backend: Literal["memory", "sqlite", "redis"] = Field(
        default="memory",
        description=(
            "Storage backend for the LangGraph node cache:\n"
            " - ``'memory'``: ``InMemoryCache`` — process-local, not shared "
            "across replicas. Cannot be combined with ``InMemorySaver``.\n"
            " - ``'sqlite'``: ``SqliteCache`` — file-based, single-process.\n"
            " - ``'redis'``: ``RedisCache`` — shared across replicas. "
            "Recommended for production."
        ),
    )
    prompt_cache: bool = Field(
        default=False,
        description=(
            "Enable provider-side prompt prefix caching. When ``True``, the "
            "static prefix of the system prompt (everything before the first "
            "``{variable}``) is annotated with ``cache_control: ephemeral`` on "
            "Anthropic models, reducing input-token costs by ~90% on cache hits."
        ),
    )
    rag_retrieval: NodeCachePolicy | None = Field(
        default_factory=lambda: NodeCachePolicy(ttl=300),
        description=(
            "Cache policy for the ``rag_retrieval`` graph node. A cache hit "
            "skips the vector-store query, returning previously retrieved chunks "
            "for the same query text. ``None`` disables caching for this node."
        ),
    )
    amem_retrieval: NodeCachePolicy | None = Field(
        default_factory=lambda: NodeCachePolicy(ttl=60),
        description=(
            "Cache policy for the ``amem_read`` graph node. A cache hit skips "
            "the knowledge-graph vector search, reusing previously retrieved "
            "notes. Write operations are never cached. ``None`` disables."
        ),
    )


# ===========================================================================
# Tool Retriever
# ===========================================================================


class ToolRetrieverConfig(BaseModel):
    """
    Controls dynamic tool selection via semantic retrieval.

    When an agent has many tools, sending all their JSON schemas on every LLM
    call wastes tokens. The tool retriever embeds each tool's description at
    compile time, then selects only the top-K most relevant tools per request
    based on cosine similarity to the user's query.

    Set ``enabled = False`` (default) to always send all tools to the LLM.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When ``False``, all tools are sent to the LLM on "
            "every call. No embedding overhead, no filtering. Recommended "
            "when the agent has fewer than ``min_tools_to_activate`` tools."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum number of tools selected per request, excluding tools "
            "listed in ``always_on``. The LLM receives at most "
            "``top_k + len(always_on)`` tool schemas per call."
        ),
    )
    min_tools_to_activate: int = Field(
        default=8,
        ge=2,
        description=(
            "Minimum total tool count required to activate dynamic retrieval. "
            "Below this threshold, the embedding and cosine-search overhead "
            "exceeds the token savings, so all tools are sent directly."
        ),
    )
    always_on: list[str] = Field(
        default_factory=list,
        description=(
            "Exact tool names that are always included in every request, never "
            "filtered by the retriever. Use for tools the user might need "
            "regardless of context, such as ``['cancel', 'help', 'handoff']``. "
            "These tools bypass similarity scoring entirely."
        ),
    )
    cache: NodeCachePolicy | None = Field(
        default_factory=lambda: NodeCachePolicy(ttl=120),
        description=(
            "Cache policy for the ``tool_retriever`` graph node. A cache hit "
            "reuses the previously selected tool set for the same query text "
            "without re-embedding. ``None`` disables caching — the retriever "
            "runs on every request."
        ),
    )


# ===========================================================================
# A-MEM — Agentic Memory
# ===========================================================================


class AMEMPromptConfig(BaseModel):
    """
    Prompt configuration for A-MEM note extraction and evolution.

    Overriding the default prompts to
    tailor memory behaviour to a specific domain.
    """

    note_extraction_prompt: str | None = Field(
        default=None,
        description=(
            "Override the default note extraction prompt. When ``None``, "
            "the built-in prompt is used. Must instruct the model to return "
            "a JSON object with fields: ``insight``, ``entities``, ``tags``, "
            "``source_summary``. Supports two placeholders:\n"
            " - ``{dialogue}``: the formatted conversation turns.\n"
            " - ``{existing_notes}``: recently retrieved notes for "
            "deduplication context."
        ),
    )
    evolution_prompt: str | None = Field(
        default=None,
        description=(
            "Override the default evolution/conflict-detection prompt. "
            "When ``None``, the built-in prompt is used. Must instruct the "
            "model to return a JSON object with fields: ``relationship``, "
            "``should_update_existing``, ``updated_insight``, ``explanation``. "
            "Supports two placeholders:\n"
            " - ``{new_insight}``: the newly extracted note insight.\n"
            " - ``{existing_insight}``: the existing note insight being compared."
        ),
    )


class AMEMConfig(BaseModel):
    """
    Agentic Memory (A-MEM) configuration.

    A-MEM builds a persistent knowledge graph on top of the standard rolling
    summary. After each agent response, an LLM analyses the exchange and
    produces a structured note — a concise insight with entities, tags, and
    links to related past notes.

    Unlike the rolling summary (linear compression of recent history), A-MEM
    captures cross-session knowledge and explicit relationships between
    concepts, users, and outcomes.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Master switch. When ``False``, no A-MEM reads or writes are "
            "performed and the ``{amem}`` placeholder expands to an empty string."
        ),
    )
    graph_backend: Literal["redis", "neo4j", "in_memory"] = Field(
        default="redis",
        description=(
            "Storage backend for the knowledge graph:\n"
            " - ``'redis'``: Redis Stack with vector-similarity search. "
            "Recommended — no extra infrastructure if Redis is already in use.\n"
            " - ``'neo4j'``: native graph database. Better for complex "
            "multi-hop relationship queries.\n"
            " - ``'in_memory'``: ephemeral dict-based store. Notes are lost "
            "on process restart. Testing only."
        ),
    )
    connection_url: str | None = Field(
        default=None,
        exclude=True,
        description=(
            "Connection URL for the A-MEM backend (e.g. ``redis://...``, "
            "``bolt://neo4j:7687``). Excluded from serialisation. When "
            "``None`` and ``graph_backend = 'redis'``, the server-managed "
            "Redis client is used."
        ),
    )
    connection_url_env: str | None = Field(
        default=None,
        description=(
            "Name of a server-side environment variable holding the connection "
            "URL. Resolved at runtime via ``os.environ[connection_url_env]``."
        ),
    )
    write_mode: Literal["async", "blocking", "disabled"] = Field(
        default="async",
        description=(
            "When and how note creation runs after an agent response:\n"
            " - ``'async'``: fire-and-forget background task. Response is "
            "delivered immediately; note creation runs in the background.\n"
            " - ``'blocking'``: ``await`` the write before releasing the "
            "processing lock. Guarantees the next request sees the new note.\n"
            " - ``'disabled'``: notes are never created. Reads from existing "
            "notes still work."
        ),
    )
    note_creation: Literal["auto", "on_tool_use", "disabled"] = Field(
        default="auto",
        description=(
            "Trigger condition for creating a new note:\n"
            " - ``'auto'``: after every completed exchange (Human → AI).\n"
            " - ``'on_tool_use'``: only when at least one tool was invoked. "
            "Reduces graph growth for simple conversational turns.\n"
            " - ``'disabled'``: no new notes are created. Useful for "
            "read-only agents consuming a shared knowledge graph."
        ),
    )
    link_extraction: bool = Field(
        default=True,
        description=(
            "When ``True``, the note-creation step also identifies semantic "
            "links between the new note and existing notes in the graph "
            "(e.g. 'refines', 'contradicts', 'extends'). Disable to reduce "
            "write latency if relationship traversal is not needed."
        ),
    )
    evolution: bool = Field(
        default=True,
        description=(
            "When ``True``, existing notes that are contradicted or superseded "
            "by a new note are updated in place. The LLM compares the new "
            "insight against similar existing notes and performs an upsert "
            "when a conflict is detected. Disable if immutability of past "
            "notes is required."
        ),
    )
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Number of notes retrieved from the knowledge graph per request. "
            "Higher values provide richer context but consume more prompt tokens."
        ),
    )
    inject_variable: str = Field(
        default="amem",
        description=(
            "Name of the system-prompt placeholder that receives retrieved "
            "notes (e.g. ``'amem'`` → ``{amem}``). Override when the default "
            "name conflicts with another placeholder."
        ),
    )
    evolution_similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine similarity score required before running the "
            "evolution LLM call against an existing note. Notes below this "
            "threshold are considered unrelated and skipped. Prevents "
            "spurious updates from weakly related notes. Range: 0.0–1.0."
        ),
    )
    scope: Literal[
        "thread",
        "thread_user",
        "agent",
        "agent_user",
        "system_agent",
        "system_agent_user",
        "system_thread",
        "system_thread_user",
    ] = Field(
        default="thread",
        description=(
            "Visibility scope of the A-MEM knowledge graph. Controls which "
            "namespace notes are written to and read from.\n\n"
            " - ``'thread'``: notes are private to the current ``session_id``. "
            "Each conversation starts with clean memory.\n"
            " - ``'thread_user'``: notes are scoped to both ``session_id`` and "
            "``user_id``. Adds explicit user isolation on top of thread isolation.\n"
            " - ``'agent'``: notes are shared across all users and all threads "
            "of this agent. Use for shared knowledge bases and FAQ accumulation.\n"
            " - ``'agent_user'``: notes are shared across all threads of this "
            "agent for the same user, isolated from other users. Recommended "
            "scope for personal assistants — the user carries memory into every "
            "new conversation.\n"
            " - ``'system_agent'``: notes are shared across all agents in this "
            "multi-agent system (identified by ``root_agent_id``) across all "
            "users and threads. All supervisor and worker agents share one pool.\n"
            " - ``'system_agent_user'``: same as ``'system_agent'`` but isolated "
            "per user. All agents in the system share memory for a given user, "
            "other users are isolated.\n"
            " - ``'system_thread'``: notes are shared across all agents in this "
            "multi-agent system within a single root conversation (identified by "
            "``root_thread_id``). Different conversations are isolated.\n"
            " - ``'system_thread_user'``: same as ``'system_thread'`` but with "
            "additional user isolation on top of root thread isolation."
        ),
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description=(
            "Model configuration for A-MEM LLM calls. "
            "Controls which model is used for note extraction and evolution."
        ),
    )
    prompt: AMEMPromptConfig = Field(
        default_factory=AMEMPromptConfig,
        description=(
            "Prompt configuration for A-MEM LLM calls. "
            "Allows overriding the default prompts."
        ),
    )


# ===========================================================================
# Memory
# ===========================================================================


class MemoryConfig(BaseModel):
    """
    Persistent memory configuration combining rolling summary and A-MEM.

    **Rolling summary** (``summarization`` + ``threshold`` + ``keep_last``):
    A compact linear digest of older turns. When the message count exceeds
    ``threshold``, oldest turns are trimmed; if ``summarization != 'disabled'``,
    an LLM produces a summary stored in ``SummaryStore`` and injected via
    ``{summary}``.

    **A-MEM** (``agentic``): A persistent cross-session knowledge graph.
    Operates independently of the rolling summary — they complement each
    other: summary provides recency, A-MEM provides depth.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Master switch for persistent memory. When ``False``, the agent is "
            "fully stateless: no checkpoint is written, no summary is maintained, "
            "and A-MEM is suppressed regardless of ``agentic.enabled``."
        ),
    )
    summarization: Literal["async", "blocking", "disabled"] = Field(
        default="disabled",
        description=(
            "LLM-based summarisation strategy after each response:\n"
            " - ``'disabled'``: old messages are trimmed but no summary is "
            "generated. Context beyond ``keep_last`` turns is lost.\n"
            " - ``'async'``: fire-and-forget background task. The next "
            "request may read a slightly stale summary.\n"
            " - ``'blocking'``: awaits summarisation before releasing the "
            "processing lock. Guarantees freshness but adds latency."
        ),
    )
    threshold: int = Field(
        default=6,
        ge=2,
        description=(
            "Number of conversation turns (counted by HumanMessage boundaries) "
            "that must accumulate before trimming and optional summarisation "
            "are triggered."
        ),
    )
    keep_last: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of most recent turns preserved verbatim after trimming. "
            "These turns remain fully visible to the LLM. Must be strictly "
            "less than ``threshold``."
        ),
    )
    agentic: AMEMConfig = Field(
        default_factory=AMEMConfig,
        description=(
            "A-MEM knowledge-graph settings. Operates as a separate memory "
            "layer on top of the rolling summary. Requires "
            "``memory.enabled = True``."
        ),
    )

    @model_validator(mode="after")
    def validate_keep_last(self):
        if self.keep_last >= self.threshold:
            raise ValueError(
                f"keep_last ({self.keep_last}) must be strictly less than "
                f"threshold ({self.threshold})."
            )
        if self.agentic.enabled and not self.enabled:
            raise ValueError("agentic.enabled=True requires memory.enabled=True.")
        return self


# ===========================================================================
# Triggers
# ===========================================================================


class TriggerConfig(BaseModel):
    """
    An event that autonomously invokes the agent without a direct user message.

    Triggers allow the agent to react to external events by injecting a
    predefined prompt as if the user had sent it. The client fires a trigger
    by sending ``{"type": "trigger", "name": "<name>"}`` over WebSocket.
    """

    name: str = Field(
        description=(
            "Unique trigger identifier within the agent. Used as the ``name`` "
            "field in the WebSocket trigger event."
        ),
    )
    type: Literal["client_event", "cron", "system"] = Field(
        description=(
            "Trigger source type:\n"
            " - ``'client_event'``: fired explicitly by the client (UI events, "
            "button clicks).\n"
            " - ``'cron'``: scheduled by the server-side scheduler.\n"
            " - ``'system'``: fired by internal infrastructure events "
            "(webhooks, queue messages)."
        ),
    )
    condition: str = Field(
        description=(
            "Human-readable description of when this trigger should fire. "
            "Used for documentation and future rule-engine evaluation."
        ),
    )
    action_prompt: str = Field(
        description=(
            "Text injected into the conversation as a ``HumanMessage`` when "
            "the trigger fires. Should instruct the agent clearly on what "
            "to do in response."
        ),
    )


# ===========================================================================
# Tool Call Retry
# ===========================================================================


class ToolCallRetryPolicy(BaseModel):
    """
    Recovery policy for tool-calling failures that happen *before* the LLM
    produces a valid AIMessage with tool_calls — i.e. provider-side schema
    validation errors, malformed function-call JSON, etc. Distinct from a
    tool that executed and returned an error (that already round-trips
    through a normal ToolMessage and needs no special handling here).
    """

    enabled: bool = Field(
        default=True,
        description="Master switch. When False, such errors propagate as before (500).",
    )
    max_retries: int = Field(
        default=2, ge=0,
        description="How many times to re-invoke the LLM after a tool-call "
                     "validation failure before giving up.",
    )
    backoff_seconds: float = Field(
        default=0.5, ge=0,
        description="Delay before each retry attempt.",
    )
    inject_feedback: bool = Field(
        default=True,
        description="Append a corrective message describing exactly what was "
                     "wrong (tool name, bad field, expected type) so the model "
                     "can self-correct instead of blindly repeating the same call.",
    )
    max_consecutive_tool_errors: int = Field(
        default=3, ge=1,
        description="Separate loop-guard: if the model keeps calling a tool "
                     "and getting back *executed* ToolMessage errors this many "
                     "times in a row (schema is fine, tool logic/args are just "
                     "wrong), stop looping and surface a graceful final answer "
                     "instead of retrying forever.",
    )


# ===========================================================================
# Tools
# ===========================================================================


class ToolParam(BaseModel):
    """Schema definition for a single parameter of a tool."""

    type: Literal["str", "int", "float", "bool"] = Field(
        description="Python primitive type of this parameter.",
    )
    description: str | None = Field(
        default=None,
        description=(
            "Human-readable description shown to the LLM. Omitting this "
            "reduces the model's ability to supply correct values."
        ),
    )
    default: Any = Field(
        default=None,
        description=(
            "Default value used when the parameter is optional and the LLM "
            "omits it. Only meaningful when ``required = False``."
        ),
    )
    required: bool = Field(
        default=True,
        description=(
            "Whether the LLM must always supply this parameter. When "
            "``False`` and ``default`` is set, the parameter is marked "
            "optional in the generated JSON Schema."
        ),
    )


class ToolConfig(BaseModel):
    """
    A tool the agent can invoke during reasoning.

    **Server tools** (``type = 'server'``): implemented server-side.
    ``AgentFactory._get_server_tool`` resolves the tool by name.

    **Client tools** (``type = 'client'``): executed client-side over
    WebSocket. The server sends ``client_tool_call`` and waits for
    ``client_tool_result`` up to ``timeout`` seconds.
    """

    name: str = Field(
        description=(
            "Tool identifier — must be unique within the agent. Used as the "
            "function name exposed to the LLM in the tools JSON Schema."
        ),
    )
    type: Literal["server", "client"] = Field(
        description=(
            "Execution location:\n"
            " - ``'server'``: runs in the backend process.\n"
            " - ``'client'``: dispatched over WebSocket; the server awaits "
            "the client's response."
        ),
    )
    description: str | None = Field(
        default=None,
        description=(
            "Explanation shown to the LLM describing what the tool does and "
            "when to use it. Critical for correct tool selection — keep it "
            "concise and action-oriented."
        ),
    )
    params: dict[str, ToolParam] = Field(
        default_factory=dict,
        description=(
            "Named parameters the tool accepts. Compiled into a Pydantic "
            "model at runtime so LangChain can generate the correct JSON Schema."
        ),
    )
    timeout: int = Field(
        default=60,
        ge=1,
        description=(
            "Maximum seconds allowed for this tool to complete.\n"
            " - **Client tools**: the server waits up to ``timeout`` seconds "
            "for a ``client_tool_result`` message over WebSocket. If none "
            "arrives, the tool returns an error string and the agent continues.\n"
            " - **Server tools**: the tool coroutine is wrapped in "
            "``asyncio.wait_for(timeout=...)``. On expiry the coroutine is "
            "cancelled and the tool returns an error string.\n"
            "Must be strictly less than ``processing_timeout``."
        ),
    )
    broadcast: BroadcastConfig = Field(
        default_factory=BroadcastConfig,
        description=(
            "Broadcast configuration for client tools. Controls which WebSocket "
            "connections receive ``client_tool_call`` and how responses are "
            "collected before the result is returned to the LLM.\n\n"
            "When ``mode='disabled'`` (default), the call is delivered only to "
            "the connection that triggered the current agent run.\n\n"
            "When any other mode is set, the call is published to all connections "
            "on this ``session_id`` across all pods via Redis pub/sub, and responses "
            "are aggregated according to the configured strategy.\n\n"
            "See ``BroadcastConfig`` and ``BroadcastMode`` for full details."
        ),
    )
    retry: ToolCallRetryPolicy | None = Field(
        default=None,
        description=(
            "Overrides AgentManifest.tool_call_retry for this specific tool when a "
            "provider-side tool-call validation failure is attributed to it. None "
            "(default) inherits the manifest-level policy. Useful for disabling "
            "retries on tools with side effects that shouldn't be attempted twice "
            "(e.g. payments), or tuning max_retries/backoff_seconds for a "
            "particularly failure-prone tool."
        ),
    )


# ===========================================================================
# Components — peripheral I/O
# ===========================================================================


class TTSConfig(BaseModel):
    """
    Text-to-Speech synthesis parameters for the Piper TTS engine.

    All fields are optional overrides of the Piper model's built-in defaults.
    TTS is activated by setting ``voice`` to a valid Piper voice model name.
    """

    voice: str | None = Field(
        default=None,
        description="Piper voice model name from the official registry.",
    )
    speaker_id: int | None = Field(
        default=None,
        description="Speaker index for multi-speaker models. Ignored for single-speaker.",
    )
    length_scale: float | None = Field(
        default=None,
        description="Speech rate multiplier. < 1.0 speeds up; > 1.0 slows down.",
    )
    noise_scale: float | None = Field(
        default=None,
        description="Generator noise controlling expressiveness and pitch variation.",
    )
    noise_w_scale: float | None = Field(
        default=None,
        description="Phoneme duration noise controlling rhythm and pause variability.",
    )
    normalize_audio: bool | None = Field(
        default=None,
        description="When True, audio samples are scaled to full amplitude range.",
    )
    volume: float | None = Field(
        default=None,
        description="Output volume multiplier. < 1.0 quieter; > 1.0 louder.",
    )


class Components(BaseModel):
    """
    Optional peripheral components attached to the agent.

    Only configure what the agent actually uses — unused components have
    no runtime cost.
    """

    tts: TTSConfig = Field(
        default_factory=TTSConfig,
        description="Text-to-Speech configuration. Activated by setting ``tts.voice``.",
    )


# ===========================================================================
# Internal helper
# ===========================================================================


def _fn_to_tool_config(fn: Any) -> ToolConfig:
    """Convert a ``@tool``-decorated client function into a ``ToolConfig``."""
    params = {k: ToolParam(**v) for k, v in fn._tool_params.items()}
    return ToolConfig(
        name=fn._tool_name,
        type="client",
        description=fn._tool_description,
        params=params,
        **fn._tool_extra,
    )


def _normalize_tool_entry(t: Any) -> list[ToolConfig]:
    """Normalise a single ``tools=[...]`` entry into 0+ ToolConfig objects."""
    if isinstance(t, ToolConfig):
        return [t]
    if isinstance(t, dict):
        return [ToolConfig(**t)]
    if isinstance(t, ToolBelt):
        return [_fn_to_tool_config(fn) for fn in t]
    if getattr(t, "_is_client_tool", False) is True:
        return [_fn_to_tool_config(t)]
    raise TypeError(
        f"Unsupported entry in 'tools': {t!r}. Expected a ToolConfig, dict, "
        f"a @tool-decorated callable, or a ToolBelt instance."
    )


# ===========================================================================
# AgentManifest — root model
# ===========================================================================


class AgentManifest(BaseModel):
    """
    Complete declarative description of an agent and its capabilities.

    The manifest is the single source of truth for how an agent behaves.
    It is submitted via ``POST /agent/deploy/{session_id}``, serialised to
    Redis, and used by ``AgentFactory`` to compile a LangGraph graph.

    Sub-agents are fully recursive — each element of ``sub_agents`` is itself
    an ``AgentManifest`` deployed independently with
    ``session_id = '{parent}_{sub.id}'``.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str = Field(
        description=(
            "Unique agent identifier within a deployment. Used to construct "
            "sub-agent thread IDs and as the key in the deploy response's "
            "``agent_ids`` mapping."
        ),
        min_length=1,
    )
    system_prompt: str = Field(
        default="",
        description=(
            "System prompt template injected before every LLM call. Supports "
            "``{time}``, ``{summary}``, ``{amem}``, ``{rag_<name>}`` "
            "placeholders. ``{summary}`` requires ``memory.summarization`` "
            "set to ``'async'`` or ``'blocking'`` (not just "
            "``memory.enabled``). Unknown placeholders produce an inline "
            "``memory.enabled``). Unknown placeholders produce an inline "
            "error note instead of crashing."
        ),
    )
    concurrency: ConcurrencyConfig = Field(
        default_factory=ConcurrencyConfig,
        description=(
            "Controls how requests are executed, including the concurrency "
            "mode and any mode-specific settings."
        )
    )
    model: list[ModelConfig] = Field(
        default_factory=list,
        max_length=10,
        description=(
            "Ordered fallback chain of LLM configurations for this agent. "
            "The first entry is tried first; if the call to it fails "
            "(connection error, provider error, etc. — but not after "
            "streaming has already started, to avoid mixed output), the "
            "next entry is tried, and so on. This chain is isolated per "
            "agent/sub-agent — it shares no state with other agents. An "
            "entry that omits base_url/model/api_key falls back to the "
            "server's global provider pool (shared, distributed rotation "
            "across all such entries system-wide), using that entry's "
            "sampling/reasoning parameters.\n\n"
            "May be left empty (the default) ONLY when "
            "model_rotation.allow_client_override=True — in that case every "
            "connecting client is required to supply its own chain via the "
            "'set_models' WebSocket event before its first message, and the "
            "server never falls back to a default chain. Leaving this empty "
            "with allow_client_override=False is a configuration error, "
            "since the agent would then have no way to obtain a model at all."
        ),
    )
    model_rotation: ModelRotationPolicy = Field(
        default_factory=ModelRotationPolicy,
        description=(
            "Controls sharing/isolation of model fallback progress across "
            "connections, and whether clients may override the model chain "
            "entirely. See ModelRotationPolicy for details."
        ),
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Persistent memory, rolling summary, and A-MEM configuration.",
    )
    rag: RAGConfig = Field(
        default_factory=RAGConfig,
        description="Retrieval-Augmented Generation configuration. Disabled by default.",
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Node-level and prompt-level caching configuration.",
    )
    tool_retriever: ToolRetrieverConfig = Field(
        default_factory=ToolRetrieverConfig,
        description=(
            "Dynamic tool selection via semantic retrieval. Disabled by "
            "default — all tools are sent on every call."
        ),
    )
    inherit_tools_from: list[str] = Field(
        default_factory=list,
        description=(
            "List of agent IDs whose tools this agent should inherit. "
            "Resolved recursively: if agent A lists B, and B lists C, then A "
            "receives tools from both B and C. On name conflicts, this "
            "agent's own tools take priority, then earlier IDs in the list "
            "win over later ones. Circular references are detected and "
            "skipped with a warning."
        ),
    )
    sub_agents: list[AgentManifest] = Field(
        default_factory=list,
        description=(
            "Nested agents grouped under this agent for organisational clarity "
            "and tool inheritance. Sub-agents are deployed independently with "
            "``session_id = '{parent_thread_id}_{sub.id}'`` and are addressable "
            "by that ID. The parent agent does NOT delegate to sub-agents "
            "automatically — delegation is an explicit, opt-in pattern: register "
            "a dedicated server tool such as ``send_task_to_{sub.id}`` that "
            "invokes the sub-agent's thread over HTTP/WebSocket and returns its "
            "response. Grouping agents here is primarily useful for "
            "``inherit_tools_from`` resolution and for deployment bookkeeping "
            "(the deploy endpoint returns all ``agent_ids`` in one response)."
        ),
    )
    tools: list[ToolConfig | dict[str, Any] | ToolBelt | Callable[..., Any]] = Field(
        default_factory=list,
        description=(
            "Tools available to this agent. Accepts ``ToolConfig`` instances, "
            "raw dicts, ``@tool``-decorated callables, and ``ToolBelt`` "
            "instances — a ``ToolBelt`` is flattened into its member tools, "
            "so passing one is equivalent to listing each of its tools "
            "individually."
        ),
    )
    tool_call_retry: ToolCallRetryPolicy = Field(
        default_factory=ToolCallRetryPolicy,
        description=(
            "Recovery policy for tool-calling failures that happen *before* a tool "
            "actually executes — the provider rejected the LLM's function-call "
            "arguments against the tool's JSON Schema (malformed/mistyped args, an "
            "explicit null where the provider doesn't allow it, etc.). Distinct from "
            "a tool that executed and returned an error string: that case already "
            "round-trips through a normal ToolMessage and the LLM can self-correct "
            "on its own next turn — no special handling needed here.\n\n"
            "On a retryable validation failure, the *same* model that failed is "
            "re-invoked (not the next entry in the ``model`` fallback chain) with a "
            "corrective system message, up to ``max_retries`` times. Falling over to "
            "the next model in ``model`` is reserved for genuine provider/connection "
            "failures — a schema-validation error never counts against provider "
            "health and never triggers rotation. If retries are exhausted, the agent "
            "responds with a graceful message instead of raising a 500.\n\n"
            "Can be overridden per tool via ``ToolConfig.retry``; ``None`` there "
            "(the default) inherits this manifest-level policy."
        ),
    )
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description=(
            "MCP servers attached to the agent. Tools, resources, and prompts "
            "provided by each server are merged with the manifest's declarative "
            "``tools``. Tool names are prefixed with ``<server_id>_`` to avoid "
            "name collisions. All loaded tools participate in "
            "``inherit_tools_from`` resolution and in the dynamic "
            "``tool_retriever`` mechanism alongside regular tools."
        ),
    )
    triggers: list[TriggerConfig] = Field(
        default_factory=list,
        description=(
            "Events that autonomously activate this agent. Fired by sending "
            "``{'type': 'trigger', 'name': '<name>'}`` over WebSocket."
        ),
    )
    permissions: AgentPermissions = Field(
        default_factory=AgentPermissions,
        description=(
            "Access control settings for this agent. Controls who can view the manifest, "
            "create threads, deploy new versions, and delete the agent.\n\n"
            "Only applicable to root agents — sub-agents inherit the root agent's "
            "deployment context and cannot have independent permissions. "
            "Specifying 'permissions' on a sub-agent raises a validation error."
        ),
    )
    components: Components | None = Field(
        default=None,
        description="Optional peripheral components (TTS; future: OCR, ASR).",
    )

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, v):
        if v is None:
            return []
        result = []
        for entry in v:
            result.extend(_normalize_tool_entry(entry))
        return result

    @model_validator(mode="after")
    def validate_model_chain_or_override(self):
        """
        `model` may only be empty when the manifest explicitly delegates
        model selection to connecting clients — otherwise there would be no
        way to ever obtain an LLM for this agent.
        """
        if not self.model and not self.model_rotation.allow_client_override:
            raise ValueError(
                "AgentManifest.model is empty and "
                "model_rotation.allow_client_override is False — this agent "
                "has no way to obtain a model chain. Either provide at least "
                "one ModelConfig in `model`, or set "
                "model_rotation.allow_client_override=True so clients can "
                "supply their own chain via the 'set_models' WS event."
            )
        return self

    @model_validator(mode="after")
    def validate_timeouts_and_tools(self):
        """Validate timeout ordering and tool constraints."""
        seen_names: set[str] = set()
        for t in self.tools:
            if t.name in seen_names:
                raise ValueError(
                    f"Duplicate tool name '{t.name}' after flattening "
                    f"ToolBelt(s)/tool list — tool names must be unique "
                    f"within an agent."
                )
            seen_names.add(t.name)

            if t.type == "client" and self.concurrency.mode == "sequential" and t.timeout >= self.concurrency.sequential.processing_timeout:
                raise ValueError(
                    f"Tool '{t.name}' timeout ({t.timeout}s) must be strictly "
                    f"less than processing_timeout ({self.concurrency.sequential.processing_timeout}s)."
                )
        return self


AgentManifest.model_rebuild()
