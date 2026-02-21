from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ===== НОВОЕ: Схема одного параметра инструмента =====
class ToolParamSchema(BaseModel):
    type: Literal["string", "integer", "float", "boolean", "array", "object"] = "string"
    description: str = ""
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None  # Допустимые значения, например ["ru", "en"]


# Конфигурация триггеров (только для главного агента)
class TriggerConfig(BaseModel):
    name: str
    type: Literal["client_event", "cron", "system"]
    condition: str
    action_prompt: str


# Конфигурация инструментов
class ToolConfig(BaseModel):
    name: str
    type: Literal["server", "client", "sub_agent"]
    description: Optional[str] = None

    # ===== ИЗМЕНЕНО: вместо Dict[str, Any] =====
    params: Dict[str, ToolParamSchema] = Field(default_factory=dict)

    sub_agent_manifest: Optional["AgentManifest"] = None


# Конфигурация суммаризации
class MemoryConfig(BaseModel):
    # Один turn = HumanMessage + все ответы до следующего HumanMessage
    # (включая AIMessage с tool_calls, ToolMessage, финальный AIMessage)
    enabled: bool = Field(default=True, description="Включить память")
    summarization: bool = Field(default=False, description="Включить суммаризацию")
    threshold: int = Field(
        default=6, description="Максимальное количество turns для хранения в памяти"
    )
    keep_last: int = Field(
        default=3,
        description="Количество turns которые останутся в памяти после переполнения threshold",
    )


class ModelConfig(BaseModel):
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = Field(
        default=None,
        description="Максимальное количество выходных токенов на ответ от модели",
    )


# ГЛАВНЫЙ МАНИФЕСТ
class AgentManifest(BaseModel):
    name: str
    system_prompt: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sub_agents: List[AgentManifest] = Field(default_factory=list)
    tools: List[ToolConfig] = Field(default_factory=list)
    triggers: List[TriggerConfig] = Field(default_factory=list)


# Для рекурсии sub_agents
AgentManifest.model_rebuild()