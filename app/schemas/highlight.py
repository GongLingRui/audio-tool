"""Highlight and note schemas."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HighlightBase(BaseModel):
    """Base highlight schema."""

    text: str
    color: Literal["yellow", "green", "blue", "pink"]
    start_offset: int = Field(..., ge=0)
    end_offset: int = Field(..., ge=0)
    chapter: str | None = None


class HighlightCreate(HighlightBase):
    """Highlight creation schema."""

    chunk_id: str | None = None
    note: str | None = None


class HighlightUpdate(BaseModel):
    """Highlight update schema."""

    color: Literal["yellow", "green", "blue", "pink"] | None = None


class Highlight(HighlightBase):
    """Highlight response schema."""

    id: str
    user_id: str
    book_id: str
    chunk_id: str | None
    note: "Note | None" = None
    created_at: datetime

    class Config:
        from_attributes = True


class NoteBase(BaseModel):
    """Base note schema."""

    content: str = Field(..., min_length=1)


class NoteCreate(NoteBase):
    """Note creation schema."""

    pass


class NoteUpdate(NoteBase):
    """Note update schema."""

    pass


class Note(NoteBase):
    """Note response schema."""

    id: str
    highlight_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Update forward reference
Highlight.model_rebuild()
