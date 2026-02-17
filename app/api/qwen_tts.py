"""
TTS API Endpoints
使用 Edge TTS - 轻量级高质量语音合成，无需下载模型
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import io

from app.services.edge_tts_service import get_edge_tts_service

logger = logging.getLogger(__name__)

router = APIRouter()


class SpeechGenerationRequest(BaseModel):
    """Request to generate speech from text."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    emotion: Optional[Dict[str, float]] = Field(None, description="Emotion parameters")
    speed: float = Field(1.0, description="Speech speed multiplier", ge=0.5, le=2.0)
    voice_id: Optional[str] = Field(None, description="Voice ID to use")


class SpeechGenerationResponse(BaseModel):
    """Response after speech generation."""
    audio_url: Optional[str] = None
    sample_rate: int
    duration: float
    format: str
    model: str
    device: str
    message: str


class VoiceCloneRequest(BaseModel):
    """Request to clone a voice."""
    voice_name: str = Field(..., description="Name for the cloned voice")
    description: Optional[str] = Field(None, description="Voice description")


class VoiceInfo(BaseModel):
    """Voice information."""
    id: str
    name: str
    language: str


class LanguageInfo(BaseModel):
    """Language information."""
    language_code: str
    language_name: str
    sample_rate: int
    model_type: str


@router.post("/generate", response_model=SpeechGenerationResponse)
async def generate_speech(request: SpeechGenerationRequest):
    """
    使用 Edge TTS 生成语音（轻量级，无需下载模型）

    Features:
    - 高质量微软神经网络TTS
    - 无需下载模型
    - 情感控制
    - 多种声音支持

    Args:
        request: Speech generation request

    Returns:
        Generated audio with metadata
    """
    try:
        tts = get_edge_tts_service()
        await tts.initialize()

        result = await tts.generate_speech(
            text=request.text,
            emotion=request.emotion,
            speed=request.speed,
        )

        # Save audio to file and generate URL
        import uuid
        from pathlib import Path
        from app.config import settings

        # 使用正确的音频目录路径
        audio_dir = settings.audio_dir / "qwen-tts"
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_id = uuid.uuid4().hex[:8]
        # Edge TTS returns MP3 format
        audio_filename = f"tts_{audio_id}.mp3"
        audio_path = audio_dir / audio_filename

        # Write audio data (MP3 format from Edge TTS)
        with open(audio_path, "wb") as f:
            f.write(result["audio"])

        logger.info(f"音频已保存到: {audio_path}")

        # Generate URL
        audio_url = f"/static/audio/qwen-tts/{audio_filename}"

        return SpeechGenerationResponse(
            audio_url=audio_url,
            sample_rate=result["sample_rate"],
            duration=result["duration"],
            format=result["format"],
            model=result["model"],
            device=result["device"],
            message="Speech generated successfully",
        )

    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-with-voice")
async def generate_speech_with_voice(
    text: str = Form(...),
    voice_sample: UploadFile = File(None),
    emotion: str = Form(None),
    speed: float = Form(1.0),
):
    """
    使用 Edge TTS 生成语音（Edge TTS不支持语音克隆，将使用默认声音）

    Args:
        text: 要合成的文本
        voice_sample: Edge TTS不支持（将被忽略）
        emotion: JSON格式的情感参数
        speed: 语速

    Returns:
        生成的音频
    """
    try:
        import json

        tts = get_edge_tts_service()
        await tts.initialize()

        # 解析情感参数
        emotion_dict = None
        if emotion:
            try:
                emotion_dict = json.loads(emotion)
            except:
                pass

        result = await tts.generate_speech(
            text=text,
            emotion=emotion_dict,
            speed=speed,
        )

        # 返回音频文件
        from fastapi.responses import Response

        return Response(
            content=result["audio"],
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=speech.mp3",
                "X-Sample-Rate": str(result["sample_rate"]),
                "X-Duration": str(result["duration"]),
            },
        )

    except Exception as e:
        logger.error(f"Error generating speech with voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clone-voice")
async def clone_voice(
    voice_name: str = Form(...),
    description: str = Form(None),
    voice_samples: List[UploadFile] = File(...),
):
    """
    Clone voice from audio samples.

    For best results:
    - Provide 5-10 audio samples
    - Each sample should be 5-15 seconds
    - Use different emotions and intonations
    - Clean audio without background noise

    Args:
        voice_name: Name for the cloned voice
        description: Voice description
        voice_samples: List of audio files

    Returns:
        Voice clone info
    """
    try:
        tts = get_qwen_tts_service()
        await tts.initialize()

        # Read all voice samples
        sample_bytes = []
        for sample in voice_samples:
            data = await sample.read()
            sample_bytes.append(data)

        result = await tts.clone_voice(
            voice_samples=sample_bytes,
            voice_name=voice_name,
            description=description,
        )

        return result

    except Exception as e:
        logger.error(f"Error cloning voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voices", response_model=List[VoiceInfo])
async def get_available_voices():
    """
    获取可用的声音列表

    Returns:
        声音信息列表
    """
    try:
        tts = get_edge_tts_service()
        await tts.initialize()

        voices = await tts.get_available_voices()

        return [VoiceInfo(**v) for v in voices]

    except Exception as e:
        logger.error(f"Error getting voices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages", response_model=List[LanguageInfo])
async def get_supported_languages():
    """
    获取支持的语言

    Returns:
        语言信息列表
    """
    try:
        tts = get_edge_tts_service()
        await tts.initialize()

        languages = await tts.get_supported_languages()

        return [LanguageInfo(**l) for l in languages]

    except Exception as e:
        logger.error(f"Error getting languages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_tts_info():
    """
    获取 TTS 服务信息

    Returns:
        服务信息
    """
    try:
        tts = get_edge_tts_service()
        await tts.initialize()

        return {
            "model": f"edge-tts-{tts.voice}",
            "device": "cloud",
            "sample_rate": 24000,
            "available_models": list(tts.CHINESE_VOICES.keys()),
            "voices": await tts.get_available_voices(),
        }

    except Exception as e:
        logger.error(f"Error getting TTS info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
