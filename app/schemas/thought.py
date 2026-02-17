"""Thought schemas."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ThoughtBase(BaseModel):
    """Base thought schema."""

    content: str = Field(..., min_length=1, max_length=10000)


class ThoughtCreate(ThoughtBase):
    """Thought creation schema."""

    book_id: str


class ThoughtUpdate(ThoughtBase):
    """Thought update schema."""

    pass


class Thought(ThoughtBase):
    """Thought response schema."""

    id: str
    user_id: str
    book_id: str
    created_at: datetime
    updated_at: datetime
    book_title: str | None = None
    book_author: str | None = None
    book_cover_url: str | None = None

    class Config:
        from_attributes = True
