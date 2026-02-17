"""Note model."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Note(Base):
    """Note model."""

    __tablename__ = "notes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    highlight_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("highlights.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    highlight = relationship("Highlight", back_populates="note")

    def __repr__(self) -> str:
        return f"<Note {self.highlight_id}>"
