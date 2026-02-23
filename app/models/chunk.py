"""Chunk model - Enhanced for new audio processing features."""
import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Chunk(Base):
    """Audio chunk model - Enhanced for audio processing features."""

    __tablename__ = "chunks"

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
    script_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("scripts.id"),
        nullable=False,
        index=True,
    )
    speaker: Mapped[str] = mapped_column(String(100), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    instruct: Mapped[str | None] = mapped_column(String(500), nullable=True)
    emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)
    section: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        index=True,
    )  # pending, processing, completed, failed
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # New fields for audio processing
    audio_task_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("audio_tasks.id"),
        nullable=True,
        index=True,
    )
    quality_metrics: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )  # {"snr_db": 25.3, "clarity": 0.85, "noise_level": 0.12}

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    project = relationship("Project", back_populates="chunks")
    script = relationship("Script", back_populates="chunks")
    audio_task = relationship("AudioTask")

    # Composite indexes for better query performance
    __table_args__ = (
        Index("ix_chunk_project_status", "project_id", "status"),
        Index("ix_chunk_project_speaker", "project_id", "speaker"),
        Index("ix_chunk_project_order", "project_id", "order_index"),
        Index("ix_chunk_script_order", "script_id", "order_index"),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.speaker}: {self.text[:30]}...>"
