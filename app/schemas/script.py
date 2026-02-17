"""Script schemas."""
from typing import Any

from pydantic import BaseModel, Field


class ScriptEntry(BaseModel):
    """Script entry schema."""

    index: int
    speaker: str = Field(..., max_length=100)
    text: str
    instruct: str | None = Field(None, max_length=500)
    emotion: str | None = Field(None, max_length=50)
    section: str | None = Field(None, max_length=255)


class ScriptBase(BaseModel):
    """Base script schema."""

    content: list[ScriptEntry]


class ScriptUpdate(ScriptBase):
    """Script update schema."""

    pass


class Script(ScriptBase):
    """Script response schema."""

    id: str
    project_id: str
    status: str
    error_message: str | None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ScriptGenerateOptions(BaseModel):
    """Script generation options."""

    system_prompt: str | None = None
    user_prompt: str | None = None
    options: dict[str, Any] | None = None


class ScriptReviewOptions(BaseModel):
    """Script review options."""

    auto_fix: bool = True
    check_rules: dict[str, bool] | None = None


class ScriptStatusResponse(BaseModel):
    """Script status response schema."""

    id: str
    status: str
    entries_count: int
    speakers: list[str]
    error_message: str | None
    created_at: str
