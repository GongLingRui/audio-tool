"""Book model."""
import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Book(Base):
    """Book model."""

    __tablename__ = "books"

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
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    author: Mapped[str | None] = mapped_column(String(255), nullable=True)
    cover_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)  # txt, pdf, epub
    total_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    projects = relationship("Project", back_populates="book", cascade="all, delete-orphan")
    highlights = relationship("Highlight", back_populates="book", cascade="all, delete-orphan")

    # Composite indexes for better query performance
    __table_args__ = (
        Index('ix_book_user_created', 'user_id', 'created_at'),
    )

    def __repr__(self) -> str:
        return f"<Book {self.title}>"
