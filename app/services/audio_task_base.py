"""Audio task base service."""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from app.database import async_session_maker


class AudioTaskBase(ABC):
    """Base class for audio task services."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def process(self, task_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Process the audio task.

        Args:
            task_id: The task ID
            params: Task parameters

        Returns:
            Processing result dictionary
        """
        pass

    @abstractmethod
    async def validate_input(self, params: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate input parameters.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass

    async def update_progress(
        self,
        task_id: str,
        progress: float,
        status: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update task progress in database.

        Args:
            task_id: The task ID
            progress: Progress value (0.0 to 1.0)
            status: Optional status update
            error_message: Optional error message
        """
        from app.models.audio_task import AudioTask
        from sqlalchemy import select, update

        async with async_session_maker() as db:
            update_data = {"progress": progress}
            if status is not None:
                update_data["status"] = status
            if error_message is not None:
                update_data["error_message"] = error_message

            await db.execute(
                update(AudioTask)
                .where(AudioTask.id == task_id)
                .values(**update_data)
            )
            await db.commit()

    async def update_task_result(
        self,
        task_id: str,
        status: str,
        output_audio_path: str | None = None,
        output_data: dict[str, Any] | None = None,
        quality_score: float | None = None,
        processing_time: float | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update task with completion results.

        Args:
            task_id: The task ID
            status: Final status
            output_audio_path: Output audio file path
            output_data: Additional output data
            quality_score: Quality score
            processing_time: Processing time in seconds
            error_message: Error message if failed
        """
        from app.models.audio_task import AudioTask
        from datetime import datetime
        from sqlalchemy import update

        async with async_session_maker() as db:
            update_data: dict[str, Any] = {
                "status": status,
                "progress": 1.0 if status == "completed" else 0.0,
            }

            if output_audio_path is not None:
                update_data["output_audio_path"] = output_audio_path
            if output_data is not None:
                update_data["output_data"] = output_data
            if quality_score is not None:
                update_data["quality_score"] = quality_score
            if processing_time is not None:
                update_data["processing_time"] = processing_time
            if error_message is not None:
                update_data["error_message"] = error_message
            if status in ("completed", "failed", "cancelled"):
                update_data["completed_at"] = datetime.utcnow()

            await db.execute(
                update(AudioTask)
                .where(AudioTask.id == task_id)
                .values(**update_data)
            )
            await db.commit()
