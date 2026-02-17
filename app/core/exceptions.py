"""Custom exceptions for the application."""
from typing import Any

from fastapi import HTTPException, status


class AppException(HTTPException):
    """Base exception for application errors."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str | None = None,
        headers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.headers = headers


class NotFoundException(AppException):
    """Resource not found exception."""

    def __init__(self, detail: str = "Resource not found") -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="NOT_FOUND",
        )


class BadRequestException(AppException):
    """Bad request exception."""

    def __init__(self, detail: str = "Bad request") -> None:
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="BAD_REQUEST",
        )


class UnauthorizedException(AppException):
    """Unauthorized exception."""

    def __init__(self, detail: str = "Unauthorized") -> None:
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED",
        )


class ForbiddenException(AppException):
    """Forbidden exception."""

    def __init__(self, detail: str = "Forbidden") -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN",
        )


class ValidationException(AppException):
    """Validation exception."""

    def __init__(self, detail: str = "Validation error") -> None:
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
        )


class TTSError(Exception):
    """TTS processing error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class LLMError(Exception):
    """LLM processing error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class AudioProcessingError(Exception):
    """Audio processing error."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
