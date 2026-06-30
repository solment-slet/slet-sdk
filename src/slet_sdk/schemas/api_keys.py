from datetime import datetime, timezone, timedelta

from pydantic import BaseModel, ConfigDict, field_validator


class CreateApiKey(BaseModel):
    name: str
    expires_at: datetime | None = None

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None

        if value.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")

        if value <= datetime.now(timezone.utc) + timedelta(minutes=1):
            raise ValueError("expires_at must be at least 1 minute in the future")

        return value


class ApiKeyWithoutKey(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    expires_at: datetime | None
    status: str


class CreateApiKeyResponse(ApiKeyWithoutKey):
    api_key: str


class GetApiKeysResponse(BaseModel):
    api_keys: list[ApiKeyWithoutKey]
