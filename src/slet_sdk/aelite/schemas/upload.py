from pydantic import BaseModel, Field


class UploadFileResponse(BaseModel):
    """Pydantic схема для ответа при загрузке файла на сервер (для дальнейшей передачи в LLM модель)"""

    file_id: str
    filename: str
    mime_type: str
    size: int
    ttl: int = Field(description="Время жизни файла в секундах")