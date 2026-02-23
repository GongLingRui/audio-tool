"""Audio task model for new audio processing features."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Float, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class AudioTaskType:
    """Audio task type enumeration."""
    VOICE_CONVERSION = "voice_conversion"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    SPEAKER_DIARIZATION = "speaker_diarization"
    SUPER_RESOLUTION = "super_resolution"
    TTS_GENERATION = "tts_generation"


class AudioTaskStatus:
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioTask(Base):
    """Audio processing task model."""

    __tablename__ = "audio_tasks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id"),
        nullable=False,
        index=True,
    )

    # Task type and status
    task_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
    )

    # Input
    input_audio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Parameters configuration
    parameters: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    # Output
    output_audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    output_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Progress and errors
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance metrics
    processing_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    # Relationships
    project = relationship("Project", back_populates="audio_tasks")

    # Composite indexes for better query performance
    __table_args__ = (
        Index("ix_task_project_type", "project_id", "task_type"),
        Index("ix_task_status_created", "status", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AudioTask {self.task_type}: {self.status}>"
