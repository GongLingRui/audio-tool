"""
LoRA Training Service
Provides LoRA fine-tuning capabilities for voice customization
"""

import logging
import os
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time
from datetime import datetime

from app.config import settings

logger = logging.getLogger(__name__)


class TrainingStatus(Enum):
    """Training status."""
    PENDING = "pending"
    PREPARING = "preparing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingConfig:
    """Configuration for LoRA training."""
    # Model settings
    base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    voice_name: str = "custom_voice"

    # Training hyperparameters
    rank: int = 32  # LoRA rank
    alpha: int = 64  # LoRA alpha
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 100

    # Data settings
    sample_rate: int = 24000
    max_duration: int = 30  # seconds per sample

    # Output settings
    output_dir: str = "/tmp/lora_checkpoints"
    save_steps: int = 500


@dataclass
class TrainingProgress:
    """Training progress information."""
    status: TrainingStatus
    current_epoch: int
    total_epochs: int
    current_step: int
    total_steps: int
    loss: float
    learning_rate: float
    eta_seconds: Optional[int]
    error_message: Optional[str]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    checkpoint_path: Optional[str] = None


class LoRATrainingService:
    """
    Service for training LoRA adapters for voice customization.

    Provides:
    - Training job management
    - Progress tracking
    - Model checkpointing
    - Validation and testing
    """

    def __init__(self):
        self.training_jobs: Dict[str, TrainingProgress] = {}
        self.checkpoint_dir = Path(settings.export_dir) / "lora_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._training_tasks: Dict[str, asyncio.Task] = {}

    async def create_training_job(
        self,
        job_id: str,
        config: TrainingConfig,
        audio_samples: List[str],
    ) -> TrainingProgress:
        """
        Create a new LoRA training job.

        Args:
            job_id: Unique identifier for the training job
            config: Training configuration
            audio_samples: List of audio file paths for training

        Returns:
            Initial training progress
        """
        # Calculate total steps based on samples and epochs
        total_steps = len(audio_samples) * config.num_epochs // config.batch_size

        progress = TrainingProgress(
            status=TrainingStatus.PENDING,
            current_epoch=0,
            total_epochs=config.num_epochs,
            current_step=0,
            total_steps=total_steps,
            loss=0.0,
            learning_rate=config.learning_rate,
            eta_seconds=None,
            error_message=None,
            started_at=None,
            completed_at=None,
            checkpoint_path=None,
        )

        self.training_jobs[job_id] = progress

        logger.info(f"Created training job {job_id} with {len(audio_samples)} samples")

        return progress

    async def start_training(
        self,
        job_id: str,
    ) -> TrainingProgress:
        """
        Start training for a job.

        This runs a simulated training process that demonstrates
        the expected behavior. In production, this would
        interface with actual ML training code.

        Args:
            job_id: Training job ID to start

        Returns:
            Updated training progress
        """
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")

        progress = self.training_jobs[job_id]

        if progress.status != TrainingStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in PENDING state")

        # Start training in background
        async def train():
            try:
                progress.status = TrainingStatus.PREPARING
                progress.started_at = datetime.now().isoformat()

                # Simulate preparation phase
                await asyncio.sleep(2)

                # Calculate steps
                steps_per_epoch = max(1, progress.total_steps // progress.total_epochs)

                progress.status = TrainingStatus.TRAINING
                progress.total_steps = steps_per_epoch * progress.total_epochs

                # Simulate training epochs
                for epoch in range(progress.total_epochs):
                    progress.current_epoch = epoch + 1

                    for step in range(steps_per_epoch):
                        if progress.status == TrainingStatus.CANCELLED:
                            return

                        progress.current_step = (epoch * steps_per_epoch) + step + 1

                        # Simulate loss decreasing
                        initial_loss = 2.0
                        target_loss = 0.1
                        total_steps = progress.total_steps
                        current_step_num = progress.current_step
                        progress.loss = initial_loss - (initial_loss - target_loss) * (current_step_num / total_steps)

                        # Calculate ETA
                        completed_ratio = current_step_num / total_steps
                        remaining_steps = total_steps - current_step_num
                        progress.eta_seconds = int(remaining_steps * 0.5)  # 0.5 sec per step

                        await asyncio.sleep(0.1)  # Simulate training time

                    # Save checkpoint after each epoch
                    checkpoint_name = f"{job_id}_epoch_{epoch + 1}.pt"
                    progress.checkpoint_path = str(self.checkpoint_dir / checkpoint_name)

                # Training completed
                progress.status = TrainingStatus.COMPLETED
                progress.completed_at = datetime.now().isoformat()
                progress.eta_seconds = 0

                # Create final checkpoint file
                final_checkpoint = self.checkpoint_dir / f"{job_id}_final.pt"
                final_checkpoint.write_text(json.dumps({
                    "job_id": job_id,
                    "status": "completed",
                    "final_loss": progress.loss,
                    "epochs": progress.total_epochs,
                    "completed_at": progress.completed_at,
                }, indent=2))
                progress.checkpoint_path = str(final_checkpoint)

                logger.info(f"Training job {job_id} completed successfully")

            except Exception as e:
                progress.status = TrainingStatus.FAILED
                progress.error_message = str(e)
                logger.error(f"Training job {job_id} failed: {e}")

        # Create background task
        task = asyncio.create_task(train())
        self._training_tasks[job_id] = task

        return progress

    async def get_training_progress(self, job_id: str) -> Optional[TrainingProgress]:
        """Get current training progress for a job."""
        return self.training_jobs.get(job_id)

    async def cancel_training(self, job_id: str) -> bool:
        """Cancel an ongoing training job."""
        if job_id not in self.training_jobs:
            return False

        progress = self.training_jobs[job_id]
        if progress.status in [TrainingStatus.TRAINING, TrainingStatus.PREPARING]:
            progress.status = TrainingStatus.CANCELLED
            logger.info(f"Cancelled training job {job_id}")
            return True

        return False

    async def list_training_jobs(
        self,
        status_filter: Optional[TrainingStatus] = None,
    ) -> List[Tuple[str, TrainingProgress]]:
        """List all training jobs, optionally filtered by status."""
        jobs = [
            (job_id, progress)
            for job_id, progress in self.training_jobs.items()
        ]

        if status_filter:
            jobs = [
                (job_id, progress)
                for job_id, progress in jobs
                if progress.status == status_filter
            ]

        return jobs

    async def get_training_checkpoint(self, job_id: str, epoch: int) -> Optional[str]:
        """Get path to training checkpoint for specific epoch."""
        checkpoint_path = self.checkpoint_dir / job_id / f"checkpoint_epoch_{epoch}.pt"

        if checkpoint_path.exists():
            return str(checkpoint_path)

        return None

    async def get_latest_checkpoint(self, job_id: str) -> Optional[str]:
        """Get path to latest checkpoint for a job."""
        # First prefer the final checkpoint file created by start_training
        final_checkpoint = self.checkpoint_dir / f"{job_id}_final.pt"
        if final_checkpoint.exists():
            return str(final_checkpoint)

        # Backwards‑compatible: also support older layout with per‑job subdirectory
        job_dir = self.checkpoint_dir / job_id
        if job_dir.exists():
            checkpoints = list(job_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                # Sort by modification time, get latest
                latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
                return str(latest)

        return None

    async def validate_training_data(
        self,
        audio_samples: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate audio samples for training.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not audio_samples:
            errors.append("没有提供训练样本")
            return False, errors

        if len(audio_samples) < 3:
            errors.append(f"训练样本数量不足：{len(audio_samples)}，建议至少 3 个")

        # Check file existence
        for sample_path in audio_samples:
            path = Path(sample_path)
            if not path.exists():
                errors.append(f"文件不存在: {sample_path}")

        return len(errors) == 0, errors

    def get_training_config_template(self) -> Dict:
        """Get template configuration for training."""
        return {
            "base_model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice_name": "my_custom_voice",
            "rank": 32,
            "alpha": 64,
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "num_epochs": 10,
            "warmup_steps": 100,
            "sample_rate": 24000,
            "max_duration": 30,
        }

    def get_training_requirements(self) -> Dict:
        """Get requirements for LoRA training."""
        return {
            "min_samples": 3,
            "recommended_samples": 10,
            "min_duration_per_sample": 5,  # seconds
            "recommended_duration_per_sample": 15,
            "total_min_duration": 30,  # seconds
            "recommended_total_duration": 120,  # seconds
            "supported_formats": [".wav", ".mp3", ".flac"],
            "sample_rate": 24000,
            "hardware": {
                "min_memory_gb": 16,
                "recommended_memory_gb": 32,
                "gpu_required": True,
                "gpu_memory_gb": 8,
            },
        }


# Singleton instance
_lora_training_service: Optional[LoRATrainingService] = None


def get_lora_training_service() -> LoRATrainingService:
    """Get the singleton LoRA training service instance."""
    global _lora_training_service
    if _lora_training_service is None:
        _lora_training_service = LoRATrainingService()
    return _lora_training_service
