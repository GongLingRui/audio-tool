"""Highlight and Note schemas."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HighlightBase(BaseModel):
    """Base highlight schema."""

    text: str = Field(..., description="Highlighted text")
    color: str = Field(default="yellow", description="Highlight color")
    note: Optional[str] = Field(None, description="Optional note")


class HighlightCreate(HighlightBase):
    """Schema for creating a highlight."""

    chapter_id: str = Field(..., description="Chapter ID")
    position: float = Field(..., description="Position in chapter")


class Highlight(HighlightBase):
    """Complete highlight schema."""

    id: str
    chapter_id: str
    position: float
    created_at: datetime

    class Config:
        from_attributes = True


class NoteBase(BaseModel):
    """Base note schema."""

    content: str = Field(..., description="Note content")
    timestamp: Optional[float] = Field(None, description="Audio timestamp")


class NoteCreate(NoteBase):
    """Schema for creating a note."""

    chunk_id: str = Field(..., description="Associated chunk ID")


class NoteUpdate(BaseModel):
    """Schema for updating a note."""

    content: Optional[str] = None
    timestamp: Optional[float] = None


class Note(NoteBase):
    """Complete note schema."""

    id: str
    chunk_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
