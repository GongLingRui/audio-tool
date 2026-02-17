"""Thought model."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Thought(Base):
    """Thought model for user's independent thoughts/ideas about books."""

    __tablename__ = "thoughts"

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
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    user = relationship("User", backref="thoughts")
    book = relationship("Book", backref="thoughts")

    def __repr__(self) -> str:
        return f"<Thought {self.id}: {self.content[:30]}...>"
