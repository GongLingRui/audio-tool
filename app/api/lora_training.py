"""
LoRA Training API Routes
Provides endpoints for LoRA model training and management
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException, BadRequestException
from app.models.project import Project
from app.schemas.common import ApiResponse
from app.services.lora_training import (
    LoRATrainingService,
    TrainingConfig,
    TrainingStatus,
    get_lora_training_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class TrainingJobCreate(BaseModel):
    """Request to create a training job."""
    voice_name: str
    rank: int = 32
    alpha: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 4


@router.get("/requirements", response_model=ApiResponse[dict])
async def get_training_requirements():
    """
    Get requirements for LoRA training.

    Returns information about:
    - Minimum and recommended sample counts
    - Audio duration requirements
    - Hardware requirements
    - Supported formats
    """
    service = get_lora_training_service()
    requirements = service.get_training_requirements()

    return ApiResponse(data=requirements)


@router.get("/config-template", response_model=ApiResponse[dict])
async def get_config_template():
    """
    Get template configuration for LoRA training.

    Returns default configuration that can be customized.
    """
    service = get_lora_training_service()
    template = service.get_training_config_template()

    return ApiResponse(data=template)


@router.post("/projects/{project_id}/train", response_model=ApiResponse[dict])
async def create_training_job(
    project_id: str,
    request: TrainingJobCreate,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """
    Create a new LoRA training job for a project.

    The job will train a custom voice adapter based on the project's
    audio samples.
    """
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    project = result.scalar_one_or_none()

    if not project:
        raise NotFoundException("Project not found")

    service = get_lora_training_service()
    job_id = f"{project_id}_{request.voice_name}"

    # Create training config
    config = TrainingConfig(
        voice_name=request.voice_name,
        rank=request.rank,
        alpha=request.alpha,
        num_epochs=request.num_epochs,
        learning_rate=request.learning_rate,
        batch_size=request.batch_size,
    )

    # Gather audio samples from project chunks
    from app.models.chunk import Chunk
    from pathlib import Path

    chunk_result = await db.execute(
        select(Chunk).where(
            Chunk.project_id == project_id,
            Chunk.status == "completed",
            Chunk.audio_path.isnot(None),
        )
    )
    chunks = chunk_result.scalars().all()

    # Collect audio samples from completed chunks
    audio_samples = []
    for chunk in chunks:
        if chunk.audio_path:
            # Verify the audio file exists
            audio_path = Path(chunk.audio_path)
            if audio_path.exists():
                # Read audio file as bytes
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                audio_samples.append({
                    "path": str(audio_path),
                    "data": audio_bytes,
                    "text": chunk.text,
                    "speaker": chunk.speaker,
                })

    if not audio_samples:
        logger.warning(f"No audio samples found for project {project_id}")

    logger.info(f"Gathered {len(audio_samples)} audio samples from {len(chunks)} completed chunks")

    # Create job with gathered audio samples
    progress = await service.create_training_job(
        job_id=job_id,
        config=config,
        audio_samples=audio_samples,
    )

    return ApiResponse(
        data={
            "job_id": job_id,
            "status": progress.status.value,
            "sample_count": len(audio_samples),
            "message": "Training job created successfully",
        }
    )


@router.post("/projects/{project_id}/start", response_model=ApiResponse[dict])
async def start_training(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Start a training job for the project."""
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_lora_training_service()

    # Find pending job for this project
    jobs = await service.list_training_jobs(status_filter=TrainingStatus.PENDING)
    project_jobs = [j for j in jobs if j[0].startswith(project_id)]

    if not project_jobs:
        raise BadRequestException("No pending training job found for this project")

    job_id = project_jobs[0][0]
    progress = await service.start_training(job_id)

    return ApiResponse(
        data={
            "job_id": job_id,
            "status": progress.status.value,
            "message": "Training started",
        }
    )


@router.get("/projects/{project_id}/progress", response_model=ApiResponse[dict])
async def get_training_progress(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get training progress for the project."""
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_lora_training_service()
    jobs = await service.list_training_jobs()

    # Find job for this project
    project_job = None
    for job_id, progress in jobs:
        if job_id.startswith(project_id):
            project_job = (job_id, progress)
            break

    if not project_job:
        return ApiResponse(
            data={
                "status": "not_started",
                "message": "No training job found for this project",
            }
        )

    job_id, progress = project_job

    # Map internal progress fields to a frontend‑friendly shape
    # elapsed_time is approximated from started_at/completed_at when available
    elapsed_time = 0
    try:
        if progress.started_at:
            from datetime import datetime

            start_dt = datetime.fromisoformat(progress.started_at)
            end_dt = datetime.fromisoformat(progress.completed_at) if progress.completed_at else datetime.now()
            elapsed_time = max(0, int((end_dt - start_dt).total_seconds()))
    except Exception:
        # Fallback to 0 if parsing fails – frontend will treat as unknown
        elapsed_time = 0

    return ApiResponse(
        data={
            "job_id": job_id,
            "status": progress.status.value,
            "current_epoch": progress.current_epoch,
            "total_epochs": progress.total_epochs,
            "current_step": progress.current_step,
            "total_steps": progress.total_steps,
            "loss": progress.loss,
            "learning_rate": progress.learning_rate,
            "elapsed_time": elapsed_time,
            "estimated_time_remaining": progress.eta_seconds,
            "checkpoint_path": progress.checkpoint_path,
            "error": progress.error_message,
        }
    )


@router.post("/projects/{project_id}/cancel", response_model=ApiResponse[dict])
async def cancel_training(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Cancel ongoing training for the project."""
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_lora_training_service()
    jobs = await service.list_training_jobs()

    # Find and cancel job for this project
    cancelled = False
    for job_id, progress in jobs:
        if job_id.startswith(project_id):
            if await service.cancel_training(job_id):
                cancelled = True
                break

    if cancelled:
        return ApiResponse(
            data={
                "message": "Training cancelled successfully",
            }
        )
    else:
        return ApiResponse(
            success=False,
            error={
                "code": "CANCEL_FAILED",
                "message": "No active training job found or could not be cancelled",
            }
        )


@router.get("/projects/{project_id}/checkpoint", response_model=ApiResponse[dict])
async def get_latest_checkpoint_metadata(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get metadata of the latest trained LoRA checkpoint for the project."""
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_lora_training_service()
    jobs = await service.list_training_jobs()

    # Find completed job for this project
    checkpoint_path = None
    for job_id, progress in jobs:
        if job_id.startswith(project_id) and progress.status == TrainingStatus.COMPLETED:
            checkpoint_path = await service.get_latest_checkpoint(job_id)
            break

    if checkpoint_path:
        return ApiResponse(
            data={
                "checkpoint_path": checkpoint_path,
                "status": "available",
            }
        )

    return ApiResponse(
        data={
            "status": "not_available",
            "message": "No trained checkpoint found for this project",
        }
    )


@router.get("/projects/{project_id}/checkpoint/download")
async def download_latest_checkpoint(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """
    Download the latest trained LoRA checkpoint for the project.

    Returns the checkpoint file as an octet‑stream response so the
    frontend can trigger a file download.
    """
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_lora_training_service()
    jobs = await service.list_training_jobs()

    checkpoint_path = None
    for job_id, progress in jobs:
        if job_id.startswith(project_id) and progress.status == TrainingStatus.COMPLETED:
            checkpoint_path = await service.get_latest_checkpoint(job_id)
            break

    if not checkpoint_path:
        raise NotFoundException("No trained checkpoint found for this project")

    return FileResponse(
        path=checkpoint_path,
        filename=f"lora_model_{project_id}.pt",
        media_type="application/octet-stream",
    )
