"""Highlight model."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Highlight(Base):
    """Text highlight model."""

    __tablename__ = "highlights"

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
    book_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("books.id"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("chunks.id"),
        nullable=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    color: Mapped[str] = mapped_column(String(20), nullable=False)  # yellow, green, blue, pink
    start_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    chapter: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)

    # Relationships
    book = relationship("Book", back_populates="highlights")
    chunk = relationship("Chunk", back_populates="highlights")
    note = relationship("Note", back_populates="highlight", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Highlight {self.color}: {self.text[:30]}...>"
