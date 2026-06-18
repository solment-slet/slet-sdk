from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, field_validator


class CreateApiKey(BaseModel):
    name: str
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime) -> datetime:
        if value <= datetime.now(value.tzinfo) + timedelta(minutes=1):
            raise ValueError("expires_at must be at least 1 minute in the future")
        return value


class ApiKeyWithoutKey(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    is_active: bool


class CreateApiKeyResponse(ApiKeyWithoutKey):
    api_key: str


class GetApiKeysResponse(BaseModel):
    api_keys: list[ApiKeyWithoutKey]
