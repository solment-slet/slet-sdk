from typing import Literal

from pydantic import BaseModel, Field

from slet_sdk.schemas import ErrorCode
from slet_sdk.schemas.client_base import ErrorResponse


# ===========================
# 401
# ===========================


class Unauthorized(ErrorResponse):
    """
    Base
    """

    status: int = 401
    error: str = ErrorCode.UNAUTHORIZED
    message: str = Field(default="Unauthorized")


class InvalidCredentials(Unauthorized):
    error: str = ErrorCode.INVALID_CREDENTIALS
    message: str = Field(
        default="Invalid Credentials", examples=["Invalid Credentials"]
    )


class TokenRefreshError(Unauthorized):
    error: str = ErrorCode.TOKEN_REFRESH_ERROR
    message: str = Field(
        default="Token Refresh Error",
        examples=["Refresh token expired", "Invalid refresh token"],
    )


class InvalidAccessToken(Unauthorized):
    error: str = ErrorCode.INVALID_ACCESS_TOKEN
    message: str = Field(default="Invalid Access Token")


class InvalidApiKey(Unauthorized):
    error: str = ErrorCode.INVALID_API_KEY
    message: str = Field(default="Invalid API Key")


class InactiveApiKey(Unauthorized):
    error: str = ErrorCode.INACTIVE_API_KEY
    message: str = Field(default="Inactive or Expired API Key")


class MissingCredentials(Unauthorized):
    error: str = ErrorCode.MISSING_CREDENTIALS
    message: str = Field(default="Missing Credentials in Headers")


# ===========================
# 403
# ===========================


class Forbidden(ErrorResponse):
    """
    Base
    """
    status: int = 403
    error: str = ErrorCode.FORBIDDEN
    message: str = Field(default="Forbidden")


# ===========================
# 404
# ===========================


class ResourceNotFound(ErrorResponse):
    """
    Base
    """

    status: int = 404
    error: str = ErrorCode.RESOURCE_NOT_FOUND
    message: str = Field(
        default="Resource Not Found",
        examples=["Resource not found", "User by id {user_id} not found"],
    )


# ===========================
# 409
# ===========================


class Conflict(ErrorResponse):
    """
    Base
    """

    status: int = 409
    error: str = ErrorCode.CONFLICT
    message: str = Field(
        default="Conflict",
        examples=["Conflict", "Email already exists"],
    )


class ApiKeyLimitExceededExtra(BaseModel):
    limit: int = Field(
        description="Maximum number of API keys per user",
        ge=0,
    )
    current_count: int = Field(
        description="The current number of API keys of the user",
        ge=0,
    )


class ApiKeyLimitExceeded(Conflict):
    error: str = ErrorCode.API_KEY_LIMIT_EXCEEDED
    message: str = Field(
        "The maximum API keys have been reached. "
        "Delete one of the existing keys to create a new one",
    )
    extra: ApiKeyLimitExceededExtra

    def __init__(self, *, limit: int, current_count: int, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = ApiKeyLimitExceededExtra(limit=limit, current_count=current_count)
        super().__init__(**kwargs)


# ===========================
# 422
# ===========================


class UnprocessableEntity(ErrorResponse):
    """
    Base
    """

    status: int = 422
    error: str = ErrorCode.UNPROCESSABLE_ENTITY
    message: str = Field(
        default="Unprocessable Entity",
        examples=["Unprocessable Entity"],
    )


class SchemaValidationErrorExtra(BaseModel):
    field: str = Field(description="Invalid field name", examples=["email"])
    detail: str = Field(
        description="Description of the validation error",
        examples=[
            "Field required",
            "value is not a valid email address: The part after the @-sign is not valid. It should have a period.",
        ],
    )


class SchemaValidationError(UnprocessableEntity):
    error: str = ErrorCode.VALIDATION_ERROR
    message: str = Field("Validation Error")
    extra: SchemaValidationErrorExtra

    def __init__(self, *, field: str, detail: str, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = SchemaValidationErrorExtra(field=field, detail=detail)
        super().__init__(**kwargs)


# ===========================
# 429
# ===========================


class TooManyRequestsExtra(BaseModel):
    retry_after: int = Field(
        description="Number of seconds to wait before retrying the request",
        examples=[60],
        ge=0,
    )


class TooManyRequests(ErrorResponse):
    """
    Base
    """

    status: int = 429
    error: str = ErrorCode.TOO_MANY_REQUESTS
    message: str = Field(
        default="Too Many Requests",
        examples=["Too Many Requests"],
    )
    extra: TooManyRequestsExtra

    def __init__(self, *, retry_after: int = 60, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = TooManyRequestsExtra(retry_after=retry_after)
        super().__init__(**kwargs)


class TooManyAttempts(TooManyRequests):
    error: str = ErrorCode.TOO_MANY_ATTEMPTS
    message: str = Field(
        default="Too Many Attempts",
        examples=["Too Many Attempts"],
    )


# ===========================
# 500
# ===========================


class InternalServerError(ErrorResponse):
    """
    Base
    """

    status: int = 500
    error: str = ErrorCode.INTERNAL_SERVER_ERROR
    message: str = Field(
        default="Internal Server Error",
        description="Description of the server error",
    )


class BadGateway(ErrorResponse):
    """
    Base
    """

    status: int = 502
    error: str = ErrorCode.BAD_GATEWAY
    message: str = Field(
        default="Bad Gateway",
    )


class ProviderConnectionError(ErrorResponse):
    status: int = 502
    error: str = ErrorCode.PROVIDER_CONNECTION_ERROR
    message: str = Field(
        default="Error Connecting to the User Provider",
    )


class ProviderTimeoutError(ErrorResponse):
    status: int = 504
    error: str = ErrorCode.PROVIDER_TIMEOUT_ERROR
    message: str = Field(
        default="Connection Timeout to the User Provider",
    )


class ProviderStatusErrorExtra(BaseModel):
    upstream_status: int
    upstream_body: dict


class ProviderStatusError(ErrorResponse):
    status: int = 502
    error: str = ErrorCode.PROVIDER_STATUS_ERROR
    message: str = Field(
        default="Bad Status From the User Provider",
    )
    extra: ProviderStatusErrorExtra

    def __init__(self, *, upstream_status: int, upstream_body: dict, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = ProviderStatusErrorExtra(
            upstream_status=upstream_status,
            upstream_body=upstream_body,
        )
        super().__init__(**kwargs)


# ===========================
# Custom (API errors unrelated to the server response)
# ===========================


class CustomError(ErrorResponse):
    """
    Base
    """

    status: Literal[599, 600] = Field(
        600, description="599 for NetworkError, 600 for anything custom errors"
    )
    error: str = ErrorCode.UNKNOWN_ERROR
    message: str = Field(default="Unknown Error")


class NetworkError(CustomError):
    """
    Any network errors on the client are used in the SDK.
    """

    status: int = 599
    error: str = ErrorCode.NETWORK_ERROR
    message: str = Field(
        default="Network Error",
        description="Network Error",
    )


class CallbackError(CustomError):
    """
    Error executing user callback
    """

    status: int = 600
    error: str = ErrorCode.CALLBACK_ERROR
