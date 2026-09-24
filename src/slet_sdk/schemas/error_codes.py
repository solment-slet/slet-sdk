from enum import Enum


class ErrorCode(str, Enum):
    """
    Каталог кодов ошибок (не HTTP кодов)
    """

    UNKNOWN_ERROR = "unknown_error"

    # 4xx
    ## 400
    BAD_REQUEST = "bad_request"

    ## 401
    UNAUTHORIZED = "unauthorized"  # base
    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_REFRESH_ERROR = "token_refresh_error"
    INVALID_ACCESS_TOKEN = "invalid_access_token"
    INVALID_API_KEY = "invalid_api_key"
    INACTIVE_API_KEY = "inactive_api_key"
    MISSING_CREDENTIALS = "missing_credentials"

    ## 403
    FORBIDDEN = "forbidden"  # base
    ACCESS_DENIED = "access_denied"

    ## 404
    RESOURCE_NOT_FOUND = "resource_not_found"  # base
    FILE_NOT_FOUND = "file_not_found"
    FILE_EXPIRED = "file_expired"
    FILE_NOT_FOUND_OR_EXPIRED = "file_not_found_or_expired"
    AGENT_NOT_FOUND = "agent_not_found"
    TRIGGER_NOT_FOUND = "trigger_not_found"

    ## 409
    CONFLICT = "conflict"  # base
    API_KEY_LIMIT_EXCEEDED = "api_key_limit_exceeded"

    # 413
    REQUEST_ENTITY_TOO_LARGE = "request_entity_too_large" # base

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

    ## 502
    BAD_GATEWAY = "bad_gateway"  # base
    PROVIDER_CONNECTION_ERROR = "provider_connection_error"
    PROVIDER_STATUS_ERROR = "provider_status_error"

    ## 503
    SERVICE_UNAVAILABLE = "service_unavailable"  # base

    ## 504
    PROVIDER_TIMEOUT_ERROR = "provider_timeout_error"

    ## >599 (Custom Errors)
    NETWORK_ERROR = "network_error"
    CALLBACK_ERROR = "callback_error"
    PROTOCOL_ERROR = "protocol_error"
    BACKGROUND_LISTENER_ERROR = "background_listener_error"

    ## Anything (Without the desired HTTP code)
    WEBSOCKET_ERROR = "websocket_error"
