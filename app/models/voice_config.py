"""Voice config model - Enhanced for new audio processing features."""
import uuid
from datetime import datetime

from sqlalchemy import Float, ForeignKey, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class VoiceConfig(Base):
    """Voice configuration model - Enhanced for new features."""

    __tablename__ = "voice_configs"

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
    speaker: Mapped[str] = mapped_column(String(100), nullable=False)
    voice_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )  # custom, clone, lora, design
    voice_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    style: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ref_audio_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    lora_model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    language: Mapped[str] = mapped_column(String(20), default="zh-CN", nullable=False)

    # New fields for audio processing
    conversion_config: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )  # For voice conversion: {"similarity": 0.85, "prosody_transfer": true}
    enhancement_config: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )  # For audio enhancement: {"denoise": true, "normalize": true, "eq_preset": "podcast"}
    training_status: Mapped[str] = mapped_column(
        String(20),
        default="not_trained",
        nullable=False,
    )  # not_trained, training, completed, failed
    training_progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    lora_adapter_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    project = relationship("Project", back_populates="voice_configs")

    def __repr__(self) -> str:
        return f"<VoiceConfig {self.speaker}: {self.voice_type}>"
