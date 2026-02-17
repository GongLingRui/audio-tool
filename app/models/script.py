"""Script model."""
import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Script(Base):
    """LLM generated script model."""

    __tablename__ = "scripts"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    project_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("projects.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    content: Mapped[dict] = mapped_column(JSON, default=list, nullable=False)  # JSON array of script entries
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
    )  # pending, reviewed, approved
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    project = relationship("Project", back_populates="script")
    chunks = relationship("Chunk", back_populates="script")

    def __repr__(self) -> str:
        return f"<Script {self.project_id}>"
