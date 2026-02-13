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
    message: str = Field("Unauthorized")


class InvalidCredentials(Unauthorized):
    error: str = ErrorCode.INVALID_CREDENTIALS
    message: str = Field("Invalid Credentials", example="Invalid Credentials")


class TokenRefreshError(Unauthorized):
    error: str = ErrorCode.TOKEN_REFRESH_ERROR
    message: str = Field(
        "Token Refresh Error",
        examples=["Refresh token expired", "Invalid refresh token"],
    )


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
        "Resource Not Found",
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
    message: str = Field("Conflict", examples=["Conflict", "Email already exists"])


# ===========================
# 422
# ===========================
class UnprocessableEntity(ErrorResponse):
    """
    Base
    """

    status: int = 422
    error: str = ErrorCode.UNPROCESSABLE_ENTITY
    message: str = Field("Unprocessable Entity", example="Unprocessable Entity")


class SchemaValidationErrorExtra(BaseModel):
    field: str = Field(..., description="Invalid field name", example="email")
    message: str = Field(
        ...,
        description="Description of the validation error",
        examples=[
            "Field required",
            "value is not a valid email address: The part after the @-sign is not valid. It should have a period.",
        ],
    )


class SchemaValidationError(UnprocessableEntity):
    error: str = ErrorCode.VALIDATION_ERROR
    message: str = Field("Validation Error", example="Validation Error")
    extra: SchemaValidationErrorExtra

    def __init__(self, *, field: str, message: str, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = SchemaValidationErrorExtra(field=field, message=message)
        super().__init__(**kwargs)


# ===========================
# 429
# ===========================
class TooManyRequestsExtra(BaseModel):
    retry_after: int = Field(
        ...,
        description="Number of seconds to wait before retrying the request",
        example=60,
        ge=0,
    )


class TooManyRequests(ErrorResponse):
    """
    Base
    """

    status: int = 429
    error: str = ErrorCode.TOO_MANY_REQUESTS
    message: str = Field("Too Many Requests", example="Too Many Requests")
    extra: TooManyRequestsExtra

    def __init__(self, *, retry_after: int = 60, **kwargs):
        kwargs.pop("extra", None)
        kwargs["extra"] = TooManyRequestsExtra(retry_after=retry_after)
        super().__init__(**kwargs)


class TooManyAttempts(TooManyRequests):
    error: str = ErrorCode.TOO_MANY_ATTEMPTS
    message: str = Field("Too Many Attempts", example="Too Many Attempts")


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
        "Internal Server Error", description="Description of the server error"
    )


# ===========================
# Custom
# ===========================
class NetworkError(ErrorResponse):
    """
    Base
    """

    status: int = 599
    error: str = ErrorCode.NETWORK_ERROR
    message: str = Field(
        "Network Error",
        description="Description of the network error, for example timeout",
    )
