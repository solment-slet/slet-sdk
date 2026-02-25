from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator


class TriggerConfig(BaseModel):
    name: str
    type: Literal["client_event", "cron", "system"]
    condition: str
    action_prompt: str


class ToolParam(BaseModel):
    type: Literal["str", "int", "float", "bool"]
    description: str = ""
    default: Any = None
    required: bool = True


class ToolConfig(BaseModel):
    name: str
    type: Literal["server", "client", "sub_agent"]
    description: Optional[str] = None
    params: Dict[str, ToolParam] = Field(default_factory=dict)
    sub_agent_manifest: Optional["AgentManifest"] = None
    # Возможность выполнения клиентских инструментов на всех клиентах. True на всех, False только на вызывающем
    broadcast: bool = Field(default=False, description="Выполнять на всех клиентах или только на вызывающем")


class MemoryConfig(BaseModel):
    enabled: bool = Field(default=True)
    summarization: bool = Field(default=False)
    threshold: int = Field(default=6)
    keep_last: int = Field(default=3)


class ModelConfig(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None


def _fn_to_tool_config(fn) -> ToolConfig:
    """Конвертирует декорированную @tool функцию в ToolConfig."""
    params = {}
    for param_name, param_info in fn._tool_params.items():
        params[param_name] = ToolParam(**param_info)

    return ToolConfig(
        name=fn._tool_name,
        type="client",
        description=fn._tool_description,
        params=params,
        **fn._tool_extra,  # ← broadcast=True и любые другие поля
    )


class AgentManifest(BaseModel):
    name: str
    system_prompt: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sub_agents: List[AgentManifest] = Field(default_factory=list)
    tools: List[ToolConfig] = Field(default_factory=list)  # принимает ToolConfig | callable
    triggers: List[TriggerConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def convert_tool_functions(cls, data: Any) -> Any:
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