from enum import Enum


class ErrorCode(str, Enum):
    """
    Каталог кодов ошибок (не HTTP кодов)
    """

    UNKNOWN_ERROR = "unknown_error"

    # 4xx
    ## 401
    UNAUTHORIZED = "unauthorized"  # base
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_REFRESH_ERROR = "token_refresh_error"

    ## 404
    RESOURCE_NOT_FOUND = "resource_not_found"  # base

    ## 409
    CONFLICT = "conflict"

    ## 422
    UNPROCESSABLE_ENTITY = "unprocessable_entity"  # base

    VALIDATION_ERROR = "validation_error"

    ## 429
    TOO_MANY_REQUESTS = "too_many_requests"  # base
    TOO_MANY_ATTEMPTS = "too_many_attempts"

    # 5xx
    ## 500
    INTERNAL_SERVER_ERROR = "internal_server_error"  # base

    ## 599 (Network Errors)
    NETWORK_ERROR = "network_error"
