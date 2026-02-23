"""Project model - Refactored for general audio content creation."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Float, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ProjectType:
    """Project type enumeration."""
    PODCAST = "podcast"
    VOICEOVER = "voiceover"
    ADVERTISEMENT = "advertisement"
    AUDIOBOOK = "audiobook"
    CUSTOM = "custom"


class Project(Base):
    """General audio project model for audio content creation."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    # Basic information
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_type: Mapped[str] = mapped_column(
        String(50),
        default="custom",
        nullable=False,
        index=True,
    )  # podcast, voiceover, advertisement, audiobook, custom

    # Project content (replaces book_id)
    source_type: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )  # text, file, transcript, script
    source_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Project status
    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        nullable=False,
        index=True,
    )  # draft, processing, completed, failed

    # Technical configuration
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Output
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_duration: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Metadata
    project_metadata: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
    )  # Can contain: {"speakers": [], "chapters": [], "tags": [], "cover_url": ""}

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    script = relationship(
        "Script",
        back_populates="project",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chunks = relationship("Chunk", back_populates="project", cascade="all, delete-orphan")
    voice_configs = relationship("VoiceConfig", back_populates="project", cascade="all, delete-orphan")
    audio_tasks = relationship("AudioTask", back_populates="project", cascade="all, delete-orphan")

    # Composite indexes for better query performance
    __table_args__ = (
        Index("ix_project_user_type", "user_id", "project_type"),
        Index("ix_project_user_status", "user_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<Project {self.name} ({self.project_type})>"
