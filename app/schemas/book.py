"""Book schemas."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class BookBase(BaseModel):
    """Base book schema."""

    title: str = Field(..., description="Book title")
    author: Optional[str] = Field(None, description="Book author")
    description: Optional[str] = Field(None, description="Book description")
    cover_url: Optional[str] = Field(None, description="Cover image URL")


class BookCreate(BookBase):
    """Schema for creating a book."""

    user_id: str = Field(..., description="User ID who owns the book")


class BookUpdate(BaseModel):
    """Schema for updating a book."""

    title: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    cover_url: Optional[str] = None


class Book(BookBase):
    """Complete book schema."""

    id: str
    user_id: str
    total_chapters: int = 0
    audio_duration: float = 0
    status: str = "draft"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BookUploadResponse(BaseModel):
    """Response schema for book upload."""

    book_id: str
    filename: str
    chapters_extracted: int
    message: str
