"""
Audio Tools API Routes
Provides audio quality analysis and recording guidelines
"""

import logging
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from app.schemas.common import ApiResponse
from app.services.audio_quality_checker import (
    AudioQualityChecker,
    get_audio_quality_checker,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/audio-quality/check", response_model=ApiResponse[dict])
async def check_audio_quality(
    file: UploadFile = File(..., description="Audio file to check"),
    detailed: bool = Form(True, description="Perform detailed analysis"),
):
    """
    Check audio quality for voice cloning suitability.

    Analyzes:
    - Duration (optimal: 30s - 5min)
    - Loudness levels
    - Dynamic range
    - Noise floor
    - Format compatibility
    - Potential clipping

    Returns a quality score and detailed recommendations.
    """
    # Save uploaded file temporarily
    temp_dir = Path("/tmp/audio_quality_checks")
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_path = temp_dir / file.filename
    with temp_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        checker = get_audio_quality_checker()
        report = await checker.check_audio_file(str(temp_path), detailed_analysis=detailed)

        return ApiResponse(data=report.to_dict())

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


@router.post("/audio-quality/check-batch", response_model=ApiResponse[dict])
async def check_audio_quality_batch(
    files: List[UploadFile] = File(..., description="Multiple audio files to check"),
):
    """
    Check quality of multiple audio files.

    Useful for batch uploads of voice cloning samples.
    Returns both individual reports and a combined assessment.
    """
    # Save uploaded files temporarily
    temp_dir = Path("/tmp/audio_quality_checks")
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    temp_paths = []

    try:
        for file in files:
            temp_path = temp_dir / file.filename
            temp_paths.append(temp_path)

            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)

            file_paths.append(str(temp_path))

        checker = get_audio_quality_checker()
        combined_report, individual_reports = await checker.check_multiple_files(file_paths)

        return ApiResponse(
            data={
                "combined": combined_report.to_dict(),
                "individual": [r.to_dict() for r in individual_reports],
                "files_checked": len(file_paths),
            }
        )

    finally:
        # Clean up temp files
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()


@router.get("/audio-quality/guidelines", response_model=ApiResponse[dict])
async def get_recording_guidelines():
    """
    Get comprehensive recording guidelines for voice cloning.

    Returns best practices for:
    - Duration and content
    - Environment and equipment
    - Recording technique
    - Technical specifications
    """
    checker = get_audio_quality_checker()
    guidelines = checker.get_recording_guidelines()

    return ApiResponse(data=guidelines)


@router.get("/audio-quality/consent", response_model=ApiResponse[dict])
async def get_consent_requirements():
    """
    Get consent and ethics requirements for voice cloning.

    Returns:
    - Required consent text
    - Agreement points
    - Usage limitations
    """
    checker = get_audio_quality_checker()
    requirements = checker.get_consent_requirements()

    return ApiResponse(data=requirements)
