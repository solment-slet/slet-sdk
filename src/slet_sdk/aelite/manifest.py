from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


class TriggerConfig(BaseModel):
    """
    Defines an event that can autonomously invoke the agent without a direct user message.

    Triggers allow the agent to react to external events (client-side UI events,
    scheduled jobs, or internal system signals) by injecting a predefined prompt
    into the conversation as if the user had sent it.
    """

    name: str = Field(
        description="Unique trigger identifier used to fire it via WebSocket: {type: 'trigger', name: '...'}."
    )
    type: Literal["client_event", "cron", "system"] = Field(
        description=(
            "Trigger source type:\n"
            "  - 'client_event': fired explicitly by the client over WebSocket.\n"
            "  - 'cron': scheduled recurring trigger (handled server-side).\n"
            "  - 'system': fired by internal infrastructure events."
        )
    )
    condition: str = Field(
        description="Human-readable description of when this trigger should fire. Used for documentation and future rule evaluation."
    )
    action_prompt: str = Field(
        description=(
            "The text injected into the conversation as a HumanMessage when this trigger fires. "
            "Instructs the agent what to do in response to the event."
        )
    )


class ToolParam(BaseModel):
    """
    Schema definition for a single parameter of a client-side tool.

    Used to dynamically build the Pydantic args schema that LangChain exposes
    to the LLM in the tool's JSON Schema — so the LLM knows what arguments to pass.
    """

    type: Literal["str", "int", "float", "bool"] = Field(
        description="Python primitive type of this parameter."
    )
    description: str | None = Field(
        default=None,
        description=(
            "Human-readable description of the parameter shown to the LLM in the tool schema. "
            "Optional — if omitted, the LLM receives no guidance for this parameter."
        )
    )
    default: Any = Field(
        default=None,
        description="Default value used when the parameter is optional and the LLM omits it."
    )
    required: bool = Field(
        default=True,
        description=(
            "Whether the LLM must always provide this parameter. "
            "If False and 'default' is set, the parameter becomes optional in the generated schema."
        )
    )


class ToolConfig(BaseModel):
    """
    Declares a tool the agent can invoke during reasoning.

    Two tool types are supported:
      - 'server': implemented server-side (e.g. get_time, search). Resolved by AgentFactory._get_server_tool.
      - 'client': executed client-side over WebSocket. The server sends a 'client_tool_call' event
                  and waits up to 300s for the client to respond with 'client_tool_result'.
    """

    name: str = Field(
        description="Tool identifier. Must be unique within the agent. Used as the function name exposed to the LLM."
    )
    type: Literal["server", "client"] = Field(
        description=(
            "Execution location:\n"
            "  - 'server': runs in the backend process.\n"
            "  - 'client': sends a WebSocket event and awaits the client's response."
        )
    )
    description: str | None = Field(
        default=None,
        description=(
            "Explains to the LLM what this tool does and when to use it. "
            "Critical for correct tool selection — keep it concise and unambiguous."
        )
    )
    params: dict[str, ToolParam] = Field(
        default_factory=dict,
        description="Named parameters this tool accepts. Dynamically compiled into a Pydantic schema for LangChain."
    )
    broadcast: bool = Field(
        default=False,
        description=(
            "Controls which WebSocket connections receive the 'client_tool_call' event:\n"
            "  - False (default): sent only to the connection that triggered the current agent run.\n"
            "  - True: broadcast to all connections currently subscribed to this thread_id.\n"
            "Use True for shared-state tools (e.g. updating a UI element visible to all participants)."
        )
    )


class MemoryConfig(BaseModel):
    """
    Controls the agent's persistent memory behaviour.

    Memory is implemented as a combination of:
      - Full message history stored in the LangGraph checkpoint (PostgreSQL).
      - A rolling summary stored separately in SummaryStore (Redis + PostgreSQL).

    When the message count exceeds 'threshold', older messages are trimmed.
    If summarization is enabled, the trimmed messages are condensed into a
    summary that is injected into the system prompt on every subsequent request.
    """

    enabled: bool = Field(
        default=True,
        description=(
            "Master switch for persistent memory. "
            "When False, the agent operates statelessly — no checkpoint is written "
            "and no summary is maintained. Each request starts with a blank slate."
        )
    )
    summarization: Literal["async", "blocking", "disabled"] = Field(
        default="disabled",
        description=(
            "Summarization strategy applied after each agent response:\n"
            "  - 'disabled': only trims old messages (no LLM summarization). "
            "Fast, but the agent loses context beyond 'keep_last' turns.\n"
            "  - 'blocking': awaits summarization before the next message can be processed. "
            "Guarantees the next request sees the updated summary. "
            "In sequential mode this extends the lock duration.\n"
            "  - 'async': summarization runs as a background task (fire-and-forget). "
            "The next request may read a slightly stale summary — acceptable for most use cases.\n\n"
            "Note: in sequential mode the processing lock is held for the full summarization "
            "duration regardless of this setting, so 'async' only meaningfully differs in parallel mode."
        )
    )
    threshold: int = Field(
        default=6,
        description=(
            "Number of conversation turns (HumanMessage boundaries) that must accumulate "
            "before trimming and optional summarization are triggered. "
            "Lower values = more frequent summarization, shorter context windows."
        )
    )
    keep_last: int = Field(
        default=3,
        description=(
            "Number of most recent turns to preserve verbatim after trimming. "
            "These messages remain in the checkpoint and are visible to the LLM in full. "
            "Must be less than 'threshold'."
        )
    )


class ModelConfig(BaseModel):
    """
    Per-agent LLM sampling parameters.

    All fields are optional. When omitted, the model_manager's global defaults are used.
    Useful for fine-tuning individual agents in a multi-agent setup
    (e.g. a low-temperature agent for structured extraction alongside a creative one).
    """

    temperature: float | None = Field(
        default=None,
        description="Sampling temperature. Lower = more deterministic. Typical range: 0.0–1.0."
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling threshold. Alternative to temperature; controls diversity."
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="Hard cap on the number of tokens the model may generate per response."
    )


class TTSConfig(BaseModel):
    """
    Text-to-Speech synthesis parameters for the Piper TTS engine.

    All fields are optional overrides. When omitted, the Piper model's built-in
    defaults are used. Only relevant if the agent's output is rendered as audio.
    """

    voice: str | None = Field(
        default=None,
        description="Piper voice model name from the official Piper model registry."
    )
    speaker_id: int | None = Field(
        default=None,
        description="Speaker index for multi-speaker Piper models. Ignored for single-speaker voices.",
        examples=[0]
    )
    length_scale: float | None = Field(
        default=None,
        description="Speech rate multiplier. < 1.0 speeds up, > 1.0 slows down.",
        ge=0.1, le=5.0,
        examples=[1.0, 0.8]
    )
    noise_scale: float | None = Field(
        default=None,
        description="Generator noise level. Higher values increase expressiveness and pitch variation.",
        ge=0.0,
        examples=[0.667]
    )
    noise_w_scale: float | None = Field(
        default=None,
        description="Phoneme duration noise. Controls rhythm and pause variability.",
        ge=0.0,
        examples=[0.8]
    )
    normalize_audio: bool | None = Field(
        default=None,
        description="Scale audio samples to the full amplitude range, preventing clipping.",
        examples=[True]
    )
    volume: float | None = Field(
        default=None,
        description="Output volume multiplier. < 1.0 quieter, > 1.0 louder.",
        ge=0.0,
        examples=[1.0, 1.5]
    )


class Components(BaseModel):
    """
    Optional peripheral components attached to the agent.

    Components extend the agent's I/O beyond text (e.g. voice output via TTS).
    Only configure what the agent actually uses — unused components have no runtime cost.
    """

    tts: TTSConfig = Field(
        default_factory=TTSConfig,
        description="Text-to-Speech configuration. Activate by setting 'tts.voice'."
    )
    # Future: OCR, ASR


def _fn_to_tool_config(fn) -> ToolConfig:
    """Converts a @tool-decorated client function into a ToolConfig instance."""
    params = {}
    for param_name, param_info in fn._tool_params.items():
        params[param_name] = ToolParam(**param_info)

    return ToolConfig(
        name=fn._tool_name,
        type="client",
        description=fn._tool_description,
        params=params,
        **fn._tool_extra,
    )


class AgentManifest(BaseModel):
    """
    Complete declarative description of an agent and its capabilities.

    The manifest is the single source of truth for how an agent behaves.
    It is submitted once via POST /agent/deploy/{thread_id}, serialised to Redis,
    and used to compile a LangGraph graph that is cached in memory per replica.

    Sub-agents are fully recursive — each sub-agent is itself an AgentManifest
    and is deployed as an independent entity with its own thread_id and WebSocket endpoint.

    Example (minimal):
        AgentManifest(
            id="assistant",
            system_prompt="You are a helpful assistant. Current time: {time}.",
        )

    Example (with memory and a client tool):
        AgentManifest(
            id="support-bot",
            system_prompt="You are a support agent. History: {summary}.",
            concurrency="parallel",
            memory=MemoryConfig(enabled=True, summarization="async", threshold=8, keep_last=3),
            tools=[
                ToolConfig(
                    name="escalate",
                    type="client",
                    description="Escalate the conversation to a human agent.",
                    params={"reason": ToolParam(type="str", description="Why escalation is needed.")},
                )
            ],
        )
    """

    id: str = Field(
        description=(
            "Unique agent identifier within a deployment. Used to construct sub-agent thread_ids "
            "and returned in the deploy response as the name→thread_id mapping key."
        )
    )
    system_prompt: str = Field(
        description=(
            "System prompt template injected before every LLM call. "
            "Supports the following format variables:\n"
            "  {time}    — current server time (always available).\n"
            "  {summary} — rolling conversation summary from SummaryStore (requires memory.enabled=True).\n"
            "Unknown variables cause a fallback with an inline error note rather than a crash."
        )
    )
    concurrency: Literal["sequential", "parallel"] = Field(
        default="sequential",
        description=(
            "Message processing strategy for this agent:\n"
            "  - 'sequential' (default): a Redis lock ensures only one message is processed at a time "
            "per thread_id, across all replicas. Safe for stateful workflows where strict "
            "message ordering matters.\n"
            "  - 'parallel': no processing lock. Multiple WebSocket connections can send messages "
            "simultaneously. Atomicity is enforced at the checkpoint write level via a short "
            "write-lock (milliseconds) and a read-modify-write merge strategy in the checkpointer. "
            "Suitable for multi-user or multi-device scenarios sharing one agent identity."
        )
    )
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="LLM sampling parameters for this agent. Falls back to model_manager defaults when unset."
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Persistent memory and summarization settings."
    )
    sub_agents: list[AgentManifest] = Field(
        default_factory=list,
        description=(
            "Nested agents that this agent can delegate to. Each sub-agent is deployed independently "
            "with thread_id = '{parent_thread_id}_{sub_agent.id}' and gets its own WebSocket endpoint. "
            "Sub-agents are compiled and registered in Redis before the parent agent."
        )
    )
    tools: list[ToolConfig] = Field(
        default_factory=list,
        description=(
            "Tools available to this agent during reasoning. Accepts either ToolConfig instances "
            "or @tool-decorated callables (automatically converted via _fn_to_tool_config). "
            "Server tools are resolved by AgentFactory; client tools are dispatched over WebSocket."
        )
    )
    triggers: list[TriggerConfig] = Field(
        default_factory=list,
        description=(
            "Events that can autonomously activate this agent without a user message. "
            "Fired by sending {type: 'trigger', name: '<trigger.name>'} over WebSocket."
        )
    )
    components: Components = Field(
        default_factory=Components,
        description="Optional peripheral components (TTS, and future: OCR, ASR)."
    )

    @model_validator(mode="before")
    @classmethod
    def convert_tool_functions(cls, data: Any) -> Any:
        """Normalises the tools list: converts @tool callables and raw dicts to ToolConfig instances."""
        if isinstance(data, dict):
            raw_tools = data.get("tools", [])
        elif hasattr(data, "tools"):
            raw_tools = data.tools
        else:
            return data

        converted = []
        for t in raw_tools:
            if hasattr(t, "_is_client_tool"):
                converted.append(_fn_to_tool_config(t))
            elif isinstance(t, dict):
                converted.append(ToolConfig(**t))
            elif isinstance(t, ToolConfig):
                converted.append(t)
            else:
                converted.append(t)

        if isinstance(data, dict):
            data["tools"] = converted
        return data


AgentManifest.model_rebuild()