"""Common response schemas."""
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""

    success: bool = True
    data: T | None = None
    error: "ErrorDetail | None" = None


class ErrorDetail(BaseModel):
    """Error detail."""

    code: str
    message: str
    details: dict[str, Any] | None = None


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


# Update forward reference
ApiResponse.model_rebuild()
