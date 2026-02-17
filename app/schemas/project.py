"""Project schemas."""
from datetime import datetime

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    """Project configuration schema."""

    tts_mode: str = "external"
    tts_url: str | None = None
    language: str = "zh-CN"
    parallel_workers: int | None = Field(None, ge=1, le=10)


class ProjectBase(BaseModel):
    """Base project schema."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ProjectCreate(ProjectBase):
    """Project creation schema."""

    book_id: str
    config: ProjectConfig | None = None


class ProjectUpdate(BaseModel):
    """Project update schema."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    config: ProjectConfig | None = None


class Project(ProjectBase):
    """Project response schema."""

    id: str
    book_id: str
    book_title: str | None = None
    status: str
    config: ProjectConfig
    audio_path: str | None
    duration: float | None
    created_at: datetime
    updated_at: datetime
    progress: "ProjectProgress | None" = None

    class Config:
        from_attributes = True


class ProjectProgress(BaseModel):
    """Project progress schema."""

    total_chunks: int
    completed_chunks: int
    percentage: float = Field(..., ge=0, le=100)


# Update forward reference
Project.model_rebuild()
