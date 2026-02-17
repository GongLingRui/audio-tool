"""
Voice Tools API Routes
Provides voice consistency checking
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.models.project import Project
from app.schemas.common import ApiResponse
from app.services.voice_consistency import (
    VoiceConsistencyService,
    get_voice_consistency_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{project_id}/voice-consistency/check", response_model=ApiResponse[dict])
async def check_voice_consistency(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """
    Check voice configuration consistency for a project.

    Analyzes:
    - Missing voice configs for speakers
    - Duplicate voice configs
    - Missing reference audio for voice cloning
    - Invalid emotion parameter ranges
    - Orphaned voice configs

    Returns a detailed report with issues and suggestions.
    """
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_voice_consistency_service()
    report = await service.check_consistency(db, project_id)

    return ApiResponse(data=report.to_dict())


@router.post("/{project_id}/voice-consistency/auto-fix", response_model=ApiResponse[dict])
async def auto_fix_voice_configs(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
    default_engine: str = Form("qwen-tts"),
):
    """
    Automatically create voice configs for missing speakers.

    Creates default voice configurations for any speakers in the script
    that don't have a voice configuration yet.
    """
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_voice_consistency_service()
    created = await service.create_missing_configs(
        db,
        project_id,
        default_engine=default_engine,
        use_defaults=True,
    )

    return ApiResponse(
        data={
            "message": f"Created {len(created)} voice configurations",
            "created_count": len(created),
            "configs": [
                {
                    "speaker": c.speaker,
                    "engine": c.engine,
                }
                for c in created
            ],
        }
    )


@router.get("/{project_id}/voice-consistency/suggest/{speaker}", response_model=ApiResponse[dict])
async def suggest_voice_config(
    project_id: str,
    speaker: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """
    Suggest voice configuration for a speaker based on their lines.

    Analyzes the speaker's text content to recommend appropriate
    emotion parameters and style settings.
    """
    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    service = get_voice_consistency_service()
    suggestion = await service.suggest_voice_config(db, project_id, speaker)

    return ApiResponse(data=suggestion)
