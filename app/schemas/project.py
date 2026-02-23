"""Project schemas - Refactored for general audio content creation."""
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
    project_type: str = Field(
        "custom",
        pattern="^(podcast|voiceover|advertisement|audiobook|custom)$",
    )
    source_type: str | None = Field(
        None,
        pattern="^(text|file|transcript|script)$",
    )
    source_content: str | None = None
    source_file_path: str | None = None
    metadata: dict = Field(default_factory=dict)


class ProjectCreate(ProjectBase):
    """Project creation schema."""

    config: ProjectConfig | None = None


class ProjectUpdate(BaseModel):
    """Project update schema."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    project_type: str | None = Field(
        None,
        pattern="^(podcast|voiceover|advertisement|audiobook|custom)$",
    )
    status: str | None = None
    config: ProjectConfig | None = None
    metadata: dict | None = None


class Project(ProjectBase):
    """Project response schema."""

    id: str
    user_id: str
    status: str
    config: ProjectConfig
    audio_path: str | None
    duration: float | None
    total_duration: float | None
    metadata: dict
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
