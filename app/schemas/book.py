"""Book schemas."""
from datetime import datetime

from pydantic import BaseModel, Field


class BookBase(BaseModel):
    """Base book schema."""

    title: str = Field(..., min_length=1, max_length=500)
    author: str | None = Field(None, max_length=255)


class BookCreate(BookBase):
    """Book creation schema."""

    pass


class BookUpdate(BaseModel):
    """Book update schema."""

    title: str | None = Field(None, min_length=1, max_length=500)
    author: str | None = Field(None, max_length=255)
    cover_url: str | None = None
    progress: float | None = Field(None, ge=0, le=1)


class Book(BookBase):
    """Book response schema."""

    id: str
    user_id: str
    cover_url: str | None
    file_type: str
    total_pages: int | None
    total_chars: int | None
    progress: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BookUploadResponse(BaseModel):
    """Book upload response schema."""

    id: str
    title: str
    author: str | None
    file_type: str
    total_chars: int | None
    status: str = "processing"


class BookContentResponse(BaseModel):
    """Book content response schema."""

    content: str
    chapters: list[dict] = []
    metadata: dict
