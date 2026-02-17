"""System configuration API routes."""
import asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
import psutil

from app.config import settings
from app.core.deps import DbDep
from app.schemas.common import ApiResponse

router = APIRouter()


# In-memory config storage (for demo purposes)
_config_store = {
    "tts": {
        "mode": settings.tts_mode,
        "url": settings.tts_url,
        "timeout": settings.tts_timeout,
        "parallel_workers": settings.tts_parallel_workers,
        "language": settings.tts_language,
    },
    "llm": {
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key,
        "model_name": settings.llm_model,
    },
    "prompts": {
        "script_generation": settings.script_generation_prompt,
        "script_review": settings.script_review_prompt,
    },
}


@router.get("", response_model=ApiResponse[dict])
async def get_config():
    """Get system configuration."""
    return ApiResponse(data=_config_store)


@router.patch("", response_model=ApiResponse[dict])
async def update_config(config_data: dict):
    """Update system configuration."""
    # Update config store (simplified for demo)
    for key, value in config_data.items():
        if key in _config_store and isinstance(value, dict):
            _config_store[key].update(value)
        elif isinstance(value, dict):
            # Nested update
            if key not in _config_store:
                _config_store[key] = {}
            _config_store[key].update(value)
        else:
            _config_store[key] = value

    return ApiResponse(data={"updated": True, "config": _config_store})


@router.get("/prompts/default", response_model=ApiResponse[dict])
async def get_default_prompts():
    """Get default prompts."""
    return ApiResponse(
        data={
            "script_generation": """
You are a professional audiobook script annotator. Your task is to analyze the given text and create a structured script with speaker labels, dialogue, and TTS instructions.

Guidelines:
1. Identify the narrator and all characters
2. Mark dialogue with the speaker's name
3. Include TTS instructions for emotion, tone, and pacing
4. Output must be valid JSON

Output format:
[
  {
    "index": 0,
    "speaker": "NARRATOR",
    "text": "The narrative text here",
    "instruct": "Calm, objective narration",
    "emotion": "neutral"
  },
  {
    "index": 1,
    "speaker": "CHARACTER_NAME",
    "text": "Character dialogue here",
    "instruct": "Emotional, questioning tone",
    "emotion": "curious"
  }
]
            """,
            "script_review": """
Review the following audiobook script for:
1. Speaker consistency
2. Text continuity
3. Emotion labeling accuracy
4. Missing dialogue tags

Fix any issues found and return the corrected script.
            """,
        }
    )


@router.get("/system/status", response_model=ApiResponse[dict])
async def get_system_status(db: DbDep = None):
    """Get system status."""
    # Get system resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd().anchor))

    # Check database connection
    db_status = "disconnected"
    try:
        if db:
            result = await db.execute(select(func.count()).select_from(text("1")))
            result.scalar()
            db_status = "connected"
    except Exception:
        pass

    # Check TTS service
    tts_status = "unknown"
    try:
        from app.services.tts_engine import TTSEngineFactory, TTSMode
        engine = TTSEngineFactory.create(TTSMode.LOCAL)
        voices = await engine.get_voices()
        if voices:
            tts_status = "connected"
    except Exception:
        tts_status = "error"

    # Check LLM service
    llm_status = "unknown"
    try:
        if _config_store["llm"]["api_key"] and _config_store["llm"]["base_url"]:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(
                    f"{_config_store['llm']['base_url']}/models",
                    headers={"Authorization": f"Bearer {_config_store['llm']['api_key']}"}
                )
                if response.status_code == 200:
                    llm_status = "connected"
                else:
                    llm_status = "unauthorized"
        else:
            llm_status = "not_configured"
    except Exception:
        llm_status = "disconnected"

    return ApiResponse(
        data={
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "tts": tts_status,
                "llm": llm_status,
                "database": db_status,
            },
            "resources": {
                "cpu_usage": round(cpu_percent, 2),
                "memory_usage": round(memory.percent, 2),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "disk_usage": round(disk.percent, 2),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
            },
        }
    )
