from enum import StrEnum


class ErrorCode(StrEnum):
    """
    Каталог кодов ошибок (не HTTP кодов)
    """
    UNKNOWN_ERROR = "unknown_error"

    # 4xx
    ## 401
    UNAUTHORIZED = "unauthorized"  # base
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_REFRESH_ERROR = "token_refresh_error"
    INVALID_ACCESS_TOKEN = "invalid_access_token"

    ## 403
    ACCESS_DENIED = "access_denied"

    ## 404
    RESOURCE_NOT_FOUND = "resource_not_found"  # base
    FILE_NOT_FOUND = "file_not_found"
    FILE_EXPIRED = "file_expired"
    FILE_NOT_FOUND_OR_EXPIRED = "file_not_found_or_expired"

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
    UPLOADS_BUILD_ERROR = "upload_build_error"

    ## >599 (Custom Errors)
    NETWORK_ERROR = "network_error"
    CALLBACK_ERROR = "callback_error"
    BACKGROUND_LISTENER_ERROR = "background_listener_error"

    ## Anything (Without the desired HTTP code)
    WEBSOCKET_ERROR = "websocket_error"
