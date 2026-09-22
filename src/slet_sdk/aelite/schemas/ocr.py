from enum import Enum

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
        default=None,
        description="Like plain_text, but all line breaks with hyphens are combined into words.",
    )
