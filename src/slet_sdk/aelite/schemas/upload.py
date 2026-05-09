from pydantic import Field
from slet_sdk.schemas import SuccessResponse


class UploadFileResponse(SuccessResponse):
    """Pydantic схема для ответа при загрузке файла на сервер (для дальнейшей передачи в LLM модель)"""
    status: int = 201
    message: str = "The file is uploaded"

    file_id: str
    filename: str
    mime_type: str
    size: int
    ttl: int = Field(description="Время жизни файла в секундах")