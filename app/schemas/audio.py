"""Audio schemas."""
from typing import Any

from pydantic import BaseModel, Field


class ChunkBase(BaseModel):
    """Base chunk schema."""

    speaker: str = Field(..., max_length=100)
    text: str
    instruct: str | None = Field(None, max_length=500)
    emotion: str | None = Field(None, max_length=50)
    section: str | None = Field(None, max_length=255)


class ChunkUpdate(BaseModel):
    """Chunk update schema."""

    text: str | None = None
    instruct: str | None = None
    speaker: str | None = None


class Chunk(ChunkBase):
    """Chunk response schema."""

    id: str
    project_id: str
    script_id: str
    status: str
    audio_path: str | None
    duration: float | None
    order_index: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ChunkGenerateBatch(BaseModel):
    """Batch audio generation request."""

    chunk_ids: list[str]
    mode: str = "parallel"
    workers: int = Field(2, ge=1, le=10)


class ChunkGenerateFast(BaseModel):
    """Fast batch generation request."""

    pass


class MergeAudioOptions(BaseModel):
    """Audio merge options."""

    pause_between_speakers: int = Field(500, ge=0, le=5000)
    pause_same_speaker: int = Field(250, ge=0, le=5000)
    output_format: str = "mp3"
    bitrate: str = "128k"


class AudioResponse(BaseModel):
    """Audio file response."""

    audio_url: str
    duration: float
    file_size: int | None = None
    format: str
    bitrate: str | None = None


class ChunkProgressResponse(BaseModel):
    """Chunk generation progress."""

    total: int
    completed: int
    processing: int
    pending: int
    percentage: float
    estimated_time_remaining: int | None = None
