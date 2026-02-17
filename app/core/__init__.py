"""Core module."""
from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import (
    AppException,
    AudioProcessingError,
    BadRequestException,
    ForbiddenException,
    LLMError,
    NotFoundException,
    TTSError,
    UnauthorizedException,
    ValidationException,
)
from app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)

__all__ = [
    "CurrentUserDep",
    "DbDep",
    "AppException",
    "NotFoundException",
    "BadRequestException",
    "UnauthorizedException",
    "ForbiddenException",
    "ValidationException",
    "TTSError",
    "LLMError",
    "AudioProcessingError",
    "create_access_token",
    "decode_access_token",
    "get_password_hash",
    "verify_password",
]
