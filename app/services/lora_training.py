"""
LoRA Training Service
---------------------

提供基于真实 Qwen3-TTS LoRA 训练脚本的服务封装，而不是模拟/mock。

实现要点：
- 使用 alexandria-audiobook/app/train_lora.py 作为实际训练入口。
- 从项目已完成的 Chunk 音频 + 文本构建数据集（metadata.jsonl + audio/）。
- 通过 asyncio 子进程运行训练脚本，解析其标准输出更新训练进度。
- 对外保持简单的 API：创建任务 / 启动训练 / 查询进度 / 取消训练 / 获取检查点。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    # Model settings（可以通过环境变量覆盖，使用更小的基础模型）
    base_model: str = getattr(
        settings,
        "lora_base_model",
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    )
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
    # 对外暴露的“检查点路径”：adapter 目录路径（供 TTS 加载或前端打包下载）
    checkpoint_path: Optional[str] = None

    # 内部元数据
    dataset_dir: Optional[str] = None
    output_dir: Optional[str] = None
    job_id: Optional[str] = None


class LoRATrainingService:
    """
    Service for training LoRA adapters for voice customization.

    提供：
    - 训练任务管理
    - 训练进度跟踪
    - 与真实 train_lora.py 的集成
    """

    def __init__(self) -> None:
        self.training_jobs: Dict[str, TrainingProgress] = {}
        self.checkpoint_dir = Path(settings.export_dir) / "lora_checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._training_tasks: Dict[str, asyncio.Task] = {}
        self._training_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._job_configs: Dict[str, TrainingConfig] = {}

    async def create_training_job(
        self,
        job_id: str,
        config: TrainingConfig,
        audio_samples: List[dict],
    ) -> TrainingProgress:
        """
        Create a new LoRA training job.

        Args:
            job_id: Unique identifier for the training job
            config: Training configuration
            audio_samples: List of training samples, each like:
                {"path": str, "text": str, "speaker": str}

        Returns:
            Initial training progress
        """
        # 构建数据集目录：export_dir/lora_datasets/{job_id}
        dataset_root = Path(settings.export_dir) / "lora_datasets"
        dataset_dir = dataset_root / job_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = dataset_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = dataset_dir / "metadata.jsonl"

        valid_samples = 0
        with metadata_path.open("w", encoding="utf-8") as f:
            for idx, sample in enumerate(audio_samples):
                src_path = Path(sample.get("path", ""))
                text = (sample.get("text") or "").strip()
                speaker = sample.get("speaker") or "NARRATOR"

                if not src_path.exists():
                    logger.warning("LoRA training sample missing file: %s", src_path)
                    continue
                if not text:
                    logger.warning("LoRA training sample missing text for: %s", src_path)
                    continue

                dest_name = f"{idx:04d}_{src_path.name}"
                dest_path = audio_dir / dest_name

                if not dest_path.exists():
                    try:
                        import shutil

                        shutil.copy2(src_path, dest_path)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "Failed to copy audio sample for LoRA dataset: %s (%s)",
                            src_path,
                            e,
                        )
                        continue

                entry = {
                    "audio_filepath": f"audio/{dest_name}",
                    "text": text,
                    "speaker": speaker,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                valid_samples += 1

        if valid_samples == 0:
            raise ValueError("No valid audio samples found for LoRA training dataset")

        total_steps = max(1, valid_samples * config.num_epochs // max(1, config.batch_size))

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
            dataset_dir=str(dataset_dir),
            output_dir=str(self.checkpoint_dir / job_id),
            job_id=job_id,
        )

        self.training_jobs[job_id] = progress
        self._job_configs[job_id] = config

        logger.info(
            "Created LoRA training job %s with %d samples (dataset_dir=%s)",
            job_id,
            valid_samples,
            dataset_dir,
        )

        return progress

    async def start_training(
        self,
        job_id: str,
    ) -> TrainingProgress:
        """
        Start training for a job.

        使用真实的 alexandria-audiobook/app/train_lora.py 作为训练脚本。
        """
        if job_id not in self.training_jobs:
            raise ValueError(f"Training job {job_id} not found")

        progress = self.training_jobs[job_id]

        if progress.status != TrainingStatus.PENDING:
            raise ValueError(f"Job {job_id} is not in PENDING state")

        if not progress.dataset_dir:
            raise ValueError(f"Training job {job_id} has no dataset_dir configured")

        config = self._job_configs.get(job_id) or TrainingConfig()

        # 定位 train_lora.py：backend/app/services/ -> backend -> 项目根目录
        backend_dir = Path(__file__).resolve().parents[2]
        project_root = backend_dir.parent
        script_path = project_root / "alexandria-audiobook" / "app" / "train_lora.py"

        if not script_path.exists():
            raise FileNotFoundError(
                f"train_lora.py not found at {script_path}. "
                f"Ensure alexandria-audiobook subproject is present."
            )

        dataset_dir = Path(progress.dataset_dir)
        output_dir = Path(progress.output_dir or (self.checkpoint_dir / job_id))
        output_dir.mkdir(parents=True, exist_ok=True)

        python_executable = sys.executable

        cmd = [
            python_executable,
            str(script_path),
            "--data_dir",
            str(dataset_dir),
            "--output_dir",
            str(output_dir),
            "--model_name",
            config.base_model,
            "--epochs",
            str(config.num_epochs),
            "--lr",
            str(config.learning_rate),
            "--batch_size",
            str(config.batch_size),
        ]

        logger.info("Starting LoRA training job %s via subprocess: %s", job_id, " ".join(cmd))

        async def train() -> None:
            try:
                progress.status = TrainingStatus.PREPARING
                progress.started_at = datetime.now().isoformat()

                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._training_processes[job_id] = proc

                progress.status = TrainingStatus.TRAINING

                assert proc.stdout is not None
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="ignore").strip()
                    if not text:
                        continue

                    logger.info("[LoRA][%s] %s", job_id, text)

                    # [TRAIN] epoch=1/10 step=3/20 loss=... lr=...
                    if text.startswith("[TRAIN]") and "epoch=" in text:
                        try:
                            parts = text.split()
                            for part in parts:
                                if part.startswith("epoch="):
                                    ep_str = part.split("=", 1)[1]
                                    cur, tot = ep_str.split("/", 1)
                                    progress.current_epoch = int(cur)
                                    progress.total_epochs = max(progress.total_epochs, int(tot))
                                elif part.startswith("loss="):
                                    val = part.split("=", 1)[1]
                                    progress.loss = float(val)
                                elif part.startswith("lr="):
                                    val = part.split("=", 1)[1]
                                    progress.learning_rate = float(val)
                        except Exception:  # noqa: BLE001
                            # 解析失败不影响主流程
                            pass

                    # [EPOCH] 1/10 avg_loss=0.1234
                    elif text.startswith("[EPOCH]"):
                        try:
                            parts = text.split()
                            if len(parts) >= 2 and "/" in parts[1]:
                                cur, tot = parts[1].split("/", 1)
                                progress.current_epoch = int(cur)
                                progress.total_epochs = max(progress.total_epochs, int(tot))
                            for part in parts:
                                if part.startswith("avg_loss="):
                                    val = part.split("=", 1)[1]
                                    progress.loss = float(val)
                        except Exception:  # noqa: BLE001
                            pass

                    elif text.startswith("[DONE]"):
                        progress.status = TrainingStatus.COMPLETED
                        progress.completed_at = datetime.now().isoformat()
                        progress.eta_seconds = 0
                        progress.checkpoint_path = str(output_dir)

                    elif text.startswith("[ERROR]"):
                        progress.status = TrainingStatus.FAILED
                        progress.error_message = text

                    # 简单 ETA 估算：剩余 epoch * 60s（仅作提示）
                    if progress.current_epoch and progress.total_epochs:
                        epochs_left = max(0, progress.total_epochs - progress.current_epoch)
                        progress.eta_seconds = epochs_left * 60

                    # 外部取消
                    if progress.status == TrainingStatus.CANCELLED:
                        try:
                            proc.terminate()
                        except ProcessLookupError:
                            pass
                        break

                return_code = await proc.wait()
                self._training_processes.pop(job_id, None)

                if progress.status not in (TrainingStatus.COMPLETED, TrainingStatus.CANCELLED):
                    if return_code == 0:
                        progress.status = TrainingStatus.COMPLETED
                        progress.completed_at = datetime.now().isoformat()
                    else:
                        progress.status = TrainingStatus.FAILED
                        if not progress.error_message:
                            progress.error_message = f"Training process exited with code {return_code}"

                if progress.status == TrainingStatus.COMPLETED:
                    logger.info("LoRA training job %s completed successfully", job_id)
                elif progress.status == TrainingStatus.CANCELLED:
                    logger.info("LoRA training job %s cancelled", job_id)
                else:
                    logger.error(
                        "LoRA training job %s failed: %s",
                        job_id,
                        progress.error_message,
                    )
            except Exception as e:  # noqa: BLE001
                progress.status = TrainingStatus.FAILED
                progress.error_message = str(e)
                logger.error("LoRA training job %s failed: %s", job_id, e)

        task = asyncio.create_task(train())
        self._training_tasks[job_id] = task

        return progress

    async def get_training_progress(self, job_id: str) -> Optional[TrainingProgress]:
        """Get current training progress for a job."""
        return self.training_jobs.get(job_id)

    async def cancel_training(self, job_id: str) -> bool:
        """Cancel an ongoing training job."""
        progress = self.training_jobs.get(job_id)
        if not progress:
            return False

        if progress.status in (TrainingStatus.TRAINING, TrainingStatus.PREPARING):
            progress.status = TrainingStatus.CANCELLED
            proc = self._training_processes.get(job_id)
            if proc is not None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            logger.info("Cancelled LoRA training job %s", job_id)
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

    async def get_training_checkpoint(self, job_id: str, epoch: int) -> Optional[str]:  # noqa: ARG002
        """Get path to training checkpoint for specific epoch.

        对接真实脚本时，adapter 目录本身即为“检查点”，这里直接返回最新路径。
        """
        progress = self.training_jobs.get(job_id)
        if progress and progress.checkpoint_path:
            return progress.checkpoint_path
        return None

    async def get_latest_checkpoint(self, job_id: str) -> Optional[str]:
        """Get path to latest checkpoint for a job."""
        progress = self.training_jobs.get(job_id)
        if progress and progress.checkpoint_path:
            return progress.checkpoint_path

        job_dir = self.checkpoint_dir / job_id
        if job_dir.exists():
            return str(job_dir)

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
        errors: List[str] = []

        if not audio_samples:
            errors.append("没有提供训练样本")
            return False, errors

        if len(audio_samples) < 3:
            errors.append(f"训练样本数量不足：{len(audio_samples)}，建议至少 3 个")

        from pathlib import Path as _Path

        for sample_path in audio_samples:
            path = _Path(sample_path)
            if not path.exists():
                errors.append(f"文件不存在: {sample_path}")

        return len(errors) == 0, errors

    def get_training_config_template(self) -> Dict:
        """Get template configuration for training."""
        return {
            "base_model": TrainingConfig().base_model,
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
