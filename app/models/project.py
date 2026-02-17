"""Project model."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Index, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Project(Base):
    """Audiobook project model."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    book_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("books.id"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        default="draft",
        nullable=False,
        index=True,
    )  # draft, processing, completed, failed
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)  # JSON field for TTS config
    audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration: Mapped[float | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    book = relationship("Book", back_populates="projects")
    script = relationship(
        "Script",
        back_populates="project",
        uselist=False,
        cascade="all, delete-orphan",
    )
    chunks = relationship("Chunk", back_populates="project", cascade="all, delete-orphan")
    voice_configs = relationship("VoiceConfig", back_populates="project", cascade="all, delete-orphan")

    # Composite indexes for better query performance
    __table_args__ = (
        Index('ix_project_book_status', 'book_id', 'status'),
    )

    def __repr__(self) -> str:
        return f"<Project {self.name}>"
