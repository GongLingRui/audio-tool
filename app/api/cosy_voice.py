"""
CosyVoice API Routes
Provides endpoints for CosyVoice 0.5B TTS and voice cloning
"""
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas.common import ApiResponse
from app.services.cosy_voice import (
    CosyVoiceEngine,
    get_cosy_voice,
    generate_with_cosy_voice,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Model Information
# =============================================================================

@router.get("/models", response_model=ApiResponse[list])
async def get_cosy_voice_models():
    """
    Get available CosyVoice models.

    Returns list of available 0.5B models with their features.
    """
    models = []
    for model_id, info in CosyVoiceEngine.MODELS.items():
        models.append({
            "id": model_id,
            "model_id": info["model_id"],
            "features": info["features"],
            "languages": info["languages"],
            "latency_ms": info.get("latency_ms"),
        })

    return ApiResponse(data=models)


@router.get("/info", response_model=ApiResponse[dict])
async def get_cosy_voice_info():
    """Get CosyVoice service information."""
    return ApiResponse(data={
        "service": "CosyVoice",
        "version": "0.5B",
        "supported_models": list(CosyVoiceEngine.MODELS.keys()),
        "default_model": "CosyVoice3-0.5B-2512",
        "features": [
            "text_to_speech",
            "voice_cloning",
            "zero_shot_cloning",
            "instruction_tts",
            "multi_lingual",
            "streaming",
        ],
        "supported_languages": [
            {"code": "zh", "name": "Chinese"},
            {"code": "en", "name": "English"},
            {"code": "ja", "name": "Japanese"},
            {"code": "yue", "name": "Cantonese"},
            {"code": "ko", "name": "Korean"},
        ],
        "download_links": {
            "huggingface": "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B",
            "modelscope": "https://modelscope.cn/models/iic/CosyVoice2-0.5B",
        },
    })


# =============================================================================
# Speakers and Languages
# =============================================================================

@router.get("/speakers", response_model=ApiResponse[list])
async def get_speakers(model: str = "CosyVoice3-0.5B-2512"):
    """Get available pre-built speakers."""
    engine = get_cosy_voice(model_name=model)
    speakers = engine.get_available_speakers()
    return ApiResponse(data=speakers)


@router.get("/languages", response_model=ApiResponse[list])
async def get_languages(model: str = "CosyVoice3-0.5B-2512"):
    """Get supported languages."""
    engine = get_cosy_voice(model_name=model)
    languages = engine.get_supported_languages()
    return ApiResponse(data=languages)


@router.get("/instructions", response_model=ApiResponse[dict])
async def get_style_instructions(model: str = "CosyVoice3-0.5B-2512"):
    """Get available style instructions."""
    engine = get_cosy_voice(model_name=model)
    instructions = engine.get_style_instructions()
    return ApiResponse(data=instructions)


# =============================================================================
# Text-to-Speech
# =============================================================================

@router.post("/generate", response_model=ApiResponse[dict])
async def generate_speech(
    text: str = Form(...),
    speaker: str = Form("zh-cn-female-1"),
    model: str = Form("CosyVoice3-0.5B-2512"),
    language: str = Form("auto"),
    speed: float = Form(1.0),
    temperature: float = Form(0.7),
    instruction: Optional[str] = Form(None),
    reference_audio: Optional[UploadFile] = File(None),
):
    """
    Generate speech using CosyVoice.

    Args:
        text: Input text to synthesize
        speaker: Speaker ID
        model: Model to use
        language: Language code (auto, zh, en, ja, yue, ko)
        speed: Speech speed (0.5 - 2.0)
        temperature: Sampling temperature (0.1 - 1.0)
        instruction: Optional style instruction
        reference_audio: Optional reference audio for voice cloning
    """
    # Validate parameters
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if not 0.5 <= speed <= 2.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.5 and 2.0")

    if not 0.1 <= temperature <= 1.0:
        raise HTTPException(status_code=400, detail="Temperature must be between 0.1 and 1.0")

    if model not in CosyVoiceEngine.MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model: {model}. Available: {list(CosyVoiceEngine.MODELS.keys())}"
        )

    try:
        # Save reference audio if provided
        reference_audio_path = None
        if reference_audio:
            temp_dir = Path(tempfile.gettempdir()) / "cosyvoice_refs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            ref_path = temp_dir / f"{uuid.uuid4().hex}_{reference_audio.filename}"

            with open(ref_path, "wb") as f:
                content = await reference_audio.read()
                f.write(content)

            reference_audio_path = str(ref_path)

        # Generate speech
        audio_data, duration = await generate_with_cosy_voice(
            text=text,
            speaker=speaker,
            reference_audio=reference_audio_path,
            instruction=instruction,
            model=model,
            speed=speed,
        )

        # Save audio to file
        output_dir = Path(tempfile.gettempdir()) / "cosyvoice_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"cosy_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / output_filename

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(data={
            "audio_path": str(output_path),
            "audio_url": f"/api/cosy-voice/audio/{output_filename}",
            "duration": duration,
            "text": text,
            "speaker": speaker,
            "model": model,
            "language": language,
            "speed": speed,
            "instruction": instruction,
            "voice_cloned": reference_audio is not None,
        })

    except Exception as e:
        logger.error(f"CosyVoice generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clone", response_model=ApiResponse[dict])
async def clone_voice(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    model: str = Form("CosyVoice3-0.5B-2512"),
    language: str = Form("auto"),
    instruction: Optional[str] = Form(None),
    speed: float = Form(1.0),
):
    """
    Clone voice from reference audio.

    Args:
        text: Text to synthesize
        reference_audio: Reference audio file (voice sample)
        model: Model to use
        language: Language code
        instruction: Optional style instruction
        speed: Speech speed
    """
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Save reference audio
        temp_dir = Path(tempfile.gettempdir()) / "cosyvoice_refs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        ref_path = temp_dir / f"{uuid.uuid4().hex}_{reference_audio.filename}"

        with open(ref_path, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        # Generate with voice cloning
        engine = get_cosy_voice(model_name=model)
        audio_data, duration = await engine.clone_voice(
            text=text,
            reference_audio_path=str(ref_path),
            language=language,
            instruction=instruction,
            speed=speed,
        )

        # Save audio
        output_dir = Path(tempfile.gettempdir()) / "cosyvoice_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"clone_{uuid.uuid4().hex[:8]}.wav"
        output_path = output_dir / output_filename

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(data={
            "audio_path": str(output_path),
            "audio_url": f"/api/cosy-voice/audio/{output_filename}",
            "duration": duration,
            "text": text,
            "reference_audio": reference_audio.filename,
            "model": model,
            "language": language,
            "instruction": instruction,
        })

    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Batch Processing
# =============================================================================

@router.post("/batch", response_model=ApiResponse[dict])
async def generate_batch(
    texts: List[str] = Form(...),
    speaker: str = Form("zh-cn-female-1"),
    model: str = Form("CosyVoice3-0.5B-2512"),
    speed: float = Form(1.0),
):
    """
    Generate speech for multiple texts.

    Returns a batch job ID that can be used to track progress.
    """
    batch_id = uuid.uuid4().hex[:12]

    # For now, generate sequentially
    results = []
    total_duration = 0.0

    for i, text in enumerate(texts):
        try:
            audio_data, duration = await generate_with_cosy_voice(
                text=text,
                speaker=speaker,
                model=model,
                speed=speed,
            )

            # Save individual audio
            output_dir = Path(tempfile.gettempdir()) / "cosyvoice_output" / batch_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"batch_{i}.wav"
            output_path = output_dir / output_filename

            with open(output_path, "wb") as f:
                f.write(audio_data)

            results.append({
                "index": i,
                "text": text,
                "audio_path": str(output_path),
                "audio_url": f"/api/cosy-voice/audio/{batch_id}/{output_filename}",
                "duration": duration,
            })
            total_duration += duration

        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            results.append({
                "index": i,
                "text": text,
                "error": str(e),
            })

    return ApiResponse(data={
        "batch_id": batch_id,
        "total_items": len(texts),
        "succeeded": sum(1 for r in results if "error" not in r),
        "failed": sum(1 for r in results if "error" in r),
        "total_duration": total_duration,
        "results": results,
    })


# =============================================================================
# Audio File Serving
# =============================================================================

@router.get("/audio/{filename}", response_class=FileResponse)
async def get_audio(filename: str):
    """Get generated audio file."""
    output_dir = Path(tempfile.gettempdir()) / "cosyvoice_output"
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


@router.get("/audio/{batch_id}/{filename}", response_class=FileResponse)
async def get_batch_audio(batch_id: str, filename: str):
    """Get batch generated audio file."""
    output_dir = Path(tempfile.gettempdir()) / "cosyvoice_output" / batch_id
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename,
    )


# =============================================================================
# Streaming TTS
# =============================================================================

@router.post("/stream")
async def generate_streaming(
    text: str = Form(...),
    speaker: str = Form("zh-cn-female-1"),
    model: str = Form("CosyVoice3-0.5B-2512"),
    language: str = Form("auto"),
):
    """
    Generate streaming audio with low latency.

    This endpoint returns streaming chunks of audio data.
    """
    from fastapi.responses import StreamingResponse

    engine = get_cosy_voice(model_name=model)

    async def audio_generator():
        try:
            async for chunk in engine.generate_streaming(
                text=text,
                speaker=speaker,
                language=language,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            raise

    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
