"""Voices API routes."""
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from pathlib import Path
import shutil

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.models.voice_config import VoiceConfig
from app.schemas.voice import (
    SpeakerParseResponse,
    Voice,
    VoiceConfig as VoiceConfigSchema,
    VoiceConfigSet,
    VoiceDesignRequest,
    VoiceDesignResponse,
    VoicePreviewRequest,
    VoicePreviewResponse,
    VoicesListResponse,
)
from app.schemas.common import ApiResponse
from app.config import settings

router = APIRouter()


@router.get("/reference", response_model=ApiResponse[dict])
async def get_voice_reference():
    """Get voice reference vocabulary for TTS instructions."""
    voice_reference_path = Path(__file__).parent.parent.parent / "app" / "static" / "VOICE_REFERENCE.md"

    if not voice_reference_path.exists():
        # Return basic reference if file doesn't exist
        return ApiResponse(
            data={
                "texture_timbre": {
                    "smooth": ["silky", "velvety", "creamy", "mellow", "buttery"],
                    "rough": ["gravelly", "raspy", "husky", "scratchy", "smoky"],
                    "resonance": ["booming", "chesty", "deep", "sonorous", "rumbled"],
                    "light": ["airy", "breathy", "feathery", "thin", "wispy"],
                },
                "emotion": {
                    "happy": ["joyful", "cheerful", "happy", "upbeat", "enthusiastic"],
                    "sad": ["melancholy", "sorrowful", "sad", "gloomy", "mournful"],
                    "calm": ["serene", "calm", "peaceful", "tranquil", "relaxed"],
                    "aggressive": ["fierce", "intense", "sharp", "biting", "hostile"],
                },
                "delivery": {
                    "staccato": "Short, detached, punchy delivery",
                    "legato": "Smooth, connected, flowing delivery",
                    "rapid_fire": "Fast, urgent, energetic delivery",
                    "measured": "Deliberate, slow, thoughtful pacing",
                    "drawl": "Lazily extended vowels, slow pacing",
                },
                "examples": [
                    {"register": "female alto", "texture": "husky, warm", "tone": "grounded"},
                    {"register": "male bass", "texture": "deep, resonant", "tone": "authoritative"},
                    {"register": "female mezzo-soprano", "texture": "silky, smooth", "tone": "gentle"},
                ],
            }
        )

    # Read and parse the voice reference file
    with open(voice_reference_path, 'r', encoding='utf-8') as f:
        content = f.read()

    return ApiResponse(data={"reference": content})


# Available voices
CUSTOM_VOICES = [
    {"id": "aiden", "name": "Aiden", "gender": "male", "language": "en-US"},
    {"id": "dylan", "name": "Dylan", "gender": "male", "language": "en-US"},
    {"id": "eric", "name": "Eric", "gender": "male", "language": "en-US"},
    {"id": "ryan", "name": "Ryan", "gender": "male", "language": "en-US"},
    {"id": "sarah", "name": "Sarah", "gender": "female", "language": "en-US"},
    {"id": "rachel", "name": "Rachel", "gender": "female", "language": "en-US"},
    {"id": "emma", "name": "Emma", "gender": "female", "language": "en-US"},
]

LORA_VOICES = [
    {"id": "builtin_watson", "name": "Watson", "gender": "male", "language": "en-US"},
]


@router.get("", response_model=ApiResponse[VoicesListResponse])
async def list_voices():
    """Get available voices."""
    return ApiResponse(
        data=VoicesListResponse(
            custom=[Voice(**v) for v in CUSTOM_VOICES],
            lora=[Voice(**v) for v in LORA_VOICES],
        )
    )


@router.get("/{project_id}/voices", response_model=ApiResponse[dict])
async def get_project_voices(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Get project voice configurations."""
    result = await db.execute(
        select(VoiceConfig).where(VoiceConfig.project_id == project_id)
    )
    voice_configs = result.scalars().all()

    voices = [VoiceConfigSchema.model_validate(vc) for vc in voice_configs]

    return ApiResponse(data={"voices": voices})


@router.post("/{project_id}/voices/parse", response_model=ApiResponse[SpeakerParseResponse])
async def parse_speakers(
    project_id: str,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Parse speakers from script."""
    from app.models.script import Script

    result = await db.execute(
        select(Script).where(Script.project_id == project_id)
    )
    script = result.scalar_one_or_none()

    if not script:
        raise NotFoundException("Script not found")

    # Extract unique speakers
    speakers = list(set(
        entry.get("speaker", "NARRATOR")
        for entry in script.content
    ))

    return ApiResponse(
        data=SpeakerParseResponse(
            speakers=speakers,
            total_entries=len(script.content),
        )
    )


@router.post("/{project_id}/voices/config", response_model=ApiResponse[dict])
async def set_voice_config(
    project_id: str,
    config_data: VoiceConfigSet,
    current_user: CurrentUserDep,
    db: DbDep,
):
    """Set voice configurations for project."""
    from app.models.project import Project

    # Verify project exists
    result = await db.execute(select(Project).where(Project.id == project_id))
    if not result.scalar_one_or_none():
        raise NotFoundException("Project not found")

    # Delete existing configs
    await db.execute(
        delete(VoiceConfig).where(VoiceConfig.project_id == project_id)
    )

    # Create new configs
    for voice_config in config_data.voices:
        vc = VoiceConfig(
            project_id=project_id,
            speaker=voice_config.speaker,
            voice_type=voice_config.voice_type,
            voice_name=voice_config.voice_name,
            style=voice_config.style,
            ref_audio_path=voice_config.ref_audio_path,
            lora_model_path=voice_config.lora_model_path,
            language=voice_config.language,
        )
        db.add(vc)

    await db.commit()

    return ApiResponse(
        data={
            "updated": True,
            "count": len(config_data.voices),
        }
    )


@router.post("/preview", response_model=ApiResponse[VoicePreviewResponse])
async def preview_voice(
    request_data: VoicePreviewRequest,
):
    """Generate voice preview using TTS engine."""
    from pathlib import Path
    from app.services.tts_engine import TTSEngineFactory, TTSMode
    import uuid

    # Generate audio using TTS engine
    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

    # Default preview text
    preview_text = request_data.text or "你好，这是一个语音预览示例。This is a voice preview sample."

    try:
        audio_data, duration = await tts_engine.generate(
            text=preview_text,
            speaker=request_data.voice_name
        )

        # Save audio file
        preview_id = str(uuid.uuid4())
        audio_dir = Path("./static/audio/previews")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"{preview_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(
            data=VoicePreviewResponse(
                audio_url=f"/static/audio/previews/{preview_id}.mp3",
                duration=duration,
            )
        )

    except Exception as e:
        # Return error response
        return ApiResponse(
            success=False,
            error={"code": "TTS_ERROR", "message": f"TTS preview failed: {str(e)}"}
        )


@router.post("/clone/upload", response_model=ApiResponse[dict])
async def upload_clone_audio(
    current_user: CurrentUserDep,
    audio: UploadFile = File(...),
    text: str = Form(...),
):
    """Upload reference audio for voice cloning."""
    import uuid
    from pathlib import Path

    # Validate audio file
    if not audio.filename:
        return ApiResponse(
            success=False,
            error={"code": "NO_FILE", "message": "No audio file provided"}
        )

    # Save file
    audio_id = str(uuid.uuid4())
    user_dir = Path("./static/uploads/voices") / current_user.id
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{audio_id}.wav"

    content = await audio.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return ApiResponse(
        data={
            "audio_path": str(file_path),
            "duration": len(content) / 48000,  # Rough estimate
        }
    )


@router.post("/design", response_model=ApiResponse[VoiceDesignResponse])
async def design_voice(
    request_data: VoiceDesignRequest,
):
    """Design voice from description.

    Analyzes the description and suggests appropriate voice parameters.
    In production, this would generate an actual preview audio.
    """
    import uuid
    from app.services.audio_processor import AudioProcessor
    from pathlib import Path

    description = request_data.description.lower()
    voice_id = f"designed_{uuid.uuid4().hex[:8]}"

    # Analyze description to infer voice characteristics
    characteristics = {
        "gender": None,
        "age_range": None,
        "pitch_adjust": 0.0,
        "speed_adjust": 1.0,
        "energy": 1.0,
        "emotion": "neutral",
        "timbre": None,
    }

    # Gender analysis
    if any(word in description for word in ["male", "man", "guy", "boy", "he", "him", "男", "男性"]):
        characteristics["gender"] = "male"
    elif any(word in description for word in ["female", "woman", "lady", "girl", "she", "her", "女", "女性"]):
        characteristics["gender"] = "female"
    else:
        characteristics["gender"] = "neutral"

    # Age analysis
    if any(word in description for word in ["young", "child", "kid", "boy", "girl", "年轻", "儿童", "小孩"]):
        characteristics["age_range"] = "young"
        characteristics["pitch_adjust"] = 2.0  # Higher pitch for young voices
    elif any(word in description for word in ["old", "elderly", "senior", "年老", "老人"]):
        characteristics["age_range"] = "senior"
        characteristics["pitch_adjust"] = -2.0  # Lower pitch for older voices
    else:
        characteristics["age_range"] = "middle-aged"

    # Emotion analysis
    if any(word in description for word in ["happy", "cheerful", "joyful", "excited", "快乐", "愉快", "兴奋"]):
        characteristics["emotion"] = "happy"
        characteristics["energy"] = 1.3
        characteristics["speed_adjust"] = 1.1
    elif any(word in description for word in ["sad", "melancholy", "sorrow", "悲伤", "忧郁"]):
        characteristics["emotion"] = "sad"
        characteristics["energy"] = 0.7
        characteristics["speed_adjust"] = 0.9
    elif any(word in description for word in ["angry", "furious", "intense", "生气", "愤怒"]):
        characteristics["emotion"] = "angry"
        characteristics["energy"] = 1.4
        characteristics["speed_adjust"] = 1.2
    elif any(word in description for word in ["calm", "peaceful", "serene", "gentle", "平静", "温和", "温柔"]):
        characteristics["emotion"] = "calm"
        characteristics["energy"] = 0.8
        characteristics["speed_adjust"] = 0.9
    elif any(word in description for word in ["energetic", "fast", "quick", "活力", "快速"]):
        characteristics["emotion"] = "energetic"
        characteristics["energy"] = 1.5
        characteristics["speed_adjust"] = 1.3

    # Timbre analysis
    if any(word in description for word in ["warm", "soft", "gentle", "温暖", "柔和"]):
        characteristics["timbre"] = "warm"
    elif any(word in description for word in ["bright", "clear", "sharp", "明亮", "清晰"]):
        characteristics["timbre"] = "bright"
    elif any(word in description for word in ["deep", "low", "heavy", "深沉", "低沉"]):
        characteristics["timbre"] = "deep"
        characteristics["pitch_adjust"] -= 2.0
    elif any(word in description for word in ["husky", "raspy", "沙哑"]):
        characteristics["timbre"] = "husky"

    # Style analysis
    if any(word in description for word in ["narrator", "storyteller", "讲述", "旁白"]):
        characteristics["style"] = "narrative"
    elif any(word in description for word in ["character", "role", "角色"]):
        characteristics["style"] = "character"

    # Generate suggested configuration
    suggested_config = {
        "voice_id": voice_id,
        "description": request_data.description,
        "characteristics": characteristics,
        "suggested_parameters": {
            "emotion": characteristics["emotion"],
            "energy": characteristics["energy"],
            "tempo": characteristics["speed_adjust"],
            "pitch": characteristics["pitch_adjust"],
            "gender": characteristics["gender"],
            "age_range": characteristics["age_range"],
            "timbre": characteristics["timbre"],
        },
        "recommended_tts_instruction": _generate_instruction_from_description(description),
    }

    # Save configuration
    design_dir = Path(settings.export_dir) / "voice_designs"
    design_dir.mkdir(parents=True, exist_ok=True)

    config_file = design_dir / f"{voice_id}.json"
    import json
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(suggested_config, f, indent=2, ensure_ascii=False)

    return ApiResponse(
        data=VoiceDesignResponse(
            preview_url=None,  # Would require TTS integration
            voice_id=voice_id,
            suggested_config=suggested_config,
            message="Voice design analysis complete",
        )
    )


def _generate_instruction_from_description(description: str) -> str:
    """Generate TTS instruction from voice description."""
    parts = []

    # Gender
    if any(word in description for word in ["male", "man", "男"]):
        parts.append("Male voice")
    elif any(word in description for word in ["female", "woman", "女"]):
        parts.append("Female voice")

    # Age
    if any(word in description for word in ["young", "年轻", "儿童"]):
        parts.append("young")
    elif any(word in description for word in ["old", "elderly", "年长", "老人"]):
        parts.append("elderly")

    # Emotion
    if any(word in description for word in ["happy", "joyful", "快乐", "愉快"]):
        parts.append("cheerful and upbeat")
    elif any(word in description for word in ["sad", "悲伤"]):
        parts.append("melancholic and soft")
    elif any(word in description for word in ["calm", "peaceful", "平静", "温和"]):
        parts.append("calm and gentle")
    elif any(word in description for word in ["energetic", "活力"]):
        parts.append("energetic and dynamic")

    # Quality
    if any(word in description for word in ["warm", "温暖"]):
        parts.append("with warmth")
    elif any(word in description for word in ["clear", "清晰"]):
        parts.append("clear and articulate")

    if not parts:
        return "Neutral narration voice"

    return " ".join(parts)
