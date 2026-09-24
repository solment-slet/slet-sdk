from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineType(str, Enum):
    classic   = "classic"    # PaddleOCR det+rec, no layout analysis — fast
    structure = "structure"  # PPStructureV3 — layout, tables, reading order
    vl = "vl"


# ─────────────────────────────────────────────────────────────────────────────
# HTTP — Classic OCR
# ─────────────────────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[int] = Field(description="[x1, y1, x2, y2] in pixels")


class OCRResponse(BaseModel):
    pipeline: PipelineType
    blocks: list[TextBlock]
    plain_text: str = Field(description="All blocks joined with newlines")
    combined_text: str | None = Field(
        None,
        description="Like plain_text, but all line breaks with hyphens are combined into words.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTTP — VL OCR
# ─────────────────────────────────────────────────────────────────────────────


class VLOCRResponse(BaseModel):
    """VL-pipeline response. `parsed` is populated only if the client sends a `response_schema`."""
    pipeline: PipelineType
    plain_text: str = Field(..., description="Raw text, markdown, or JSON string returned by the model")
    combined_text: str | None = Field(
        None,
        description="Like plain_text, but all line breaks with hyphens are combined into words.",
    )
    parsed: dict[str, Any] | None = Field(
        None,
        description=(
            "Parsed JSON object. Populated only if a `response_schema`"
            "was provided and the model's response is valid."
        )
    )
