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
    type: Literal["server", "client"]
    description: Optional[str] = None
    params: Dict[str, ToolParam] = Field(default_factory=dict)
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


class TTSConfig(BaseModel):
    voice: Optional[str] = Field(
        default=None,
        description="Название модели голоса из официального реестра Piper."
    )

    speaker_id: Optional[int] = Field(
        default=None,
        description="ID диктора (только для мульти-спикер моделей).",
        examples=[0]
    )

    length_scale: Optional[float] = Field(
        default=None,
        description="Скорость речи. < 1.0 быстрее, > 1.0 медленнее.",
        ge=0.1, le=5.0,
        examples=[1.0, 0.8]
    )

    noise_scale: Optional[float] = Field(
        default=None,
        description="Количество добавляемого шума генератора (эмоциональность/вариативность).",
        ge=0.0,
        examples=[0.667]
    )

    noise_w_scale: Optional[float] = Field(
        default=None,
        description="Количество шума длительности фонем (вариативность пауз и ритма).",
        ge=0.0,
        examples=[0.8]
    )

    normalize_audio: Optional[bool] = Field(
        default=None,
        description="Включить/выключить масштабирование семплов аудио для заполнения полного диапазона (предотвращает перегруз).",
        examples=[True]
    )

    volume: Optional[float] = Field(
        default=None,
        description="Множитель громкости. < 1.0 тише, > 1.0 громче.",
        ge=0.0,
        examples=[1.0, 1.5]
    )


class Components(BaseModel):
    tts: TTSConfig = Field(default_factory=TTSConfig)
    # OCR, ASR


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
    id: str
    system_prompt: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    sub_agents: List[AgentManifest] = Field(default_factory=list)
    tools: List[ToolConfig] = Field(default_factory=list)  # принимает ToolConfig | callable
    triggers: List[TriggerConfig] = Field(default_factory=list)
    components: Components = Field(default_factory=Components)

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