"""Voice styling and emotion control API routes."""
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid

from app.core.deps import CurrentUserDep, DbDep
from app.core.exceptions import NotFoundException
from app.schemas.voice_emotion import (
    EmotionParameters,
    EmotionPreset,
    VoiceStyle,
    VoiceStylingRequest,
    VoiceConversionRequest,
    BatchVoiceCloneRequest,
    TTSLanguageConfig,
    SpeechEnhancementRequest,
)
from app.schemas.common import ApiResponse

# Lazy import to avoid missing dependencies
def _get_audio_processor():
    from app.services.audio_processor import AudioProcessor
    return AudioProcessor

router = APIRouter()


# Built-in emotion presets
EMOTION_PRESETS = [
    EmotionPreset(
        id="neutral",
        name="Neutral",
        description="Calm, objective delivery",
        emotion=EmotionParameters(neutral=1.0, energy=1.0),
        example_instruct="Neutral, even narration."
    ),
    EmotionPreset(
        id="happy",
        name="Happy/Joyful",
        description="Cheerful and upbeat",
        emotion=EmotionParameters(happiness=0.8, energy=1.3, tempo=1.1),
        example_instruct="Joyful, cheerful tone with bright energy."
    ),
    EmotionPreset(
        id="sad",
        name="Sad/Melancholy",
        description="Sorrowful and subdued",
        emotion=EmotionParameters(sadness=0.8, energy=0.7, tempo=0.9),
        example_instruct="Melancholy, quiet sorrow with measured pacing."
    ),
    EmotionPreset(
        id="angry",
        name="Angry/Intense",
        description="Frustrated and aggressive",
        emotion=EmotionParameters(anger=0.8, energy=1.4, tempo=1.2, volume=1.2),
        example_instruct="Fierce, intense anger with sharp delivery."
    ),
    EmotionPreset(
        id="fearful",
        name="Fearful/Anxious",
        description="Scared and hesitant",
        emotion=EmotionParameters(fear=0.8, energy=0.8, tempo=1.1),
        example_instruct="Anxious, trembling fear with urgent delivery."
    ),
    EmotionPreset(
        id="surprised",
        name="Surprised/Shocked",
        description="Shocked and amazed",
        emotion=EmotionParameters(surprise=0.8, energy=1.2),
        example_instruct="Shocked, amazed tone with wide-eyed wonder."
    ),
    EmotionPreset(
        id="romantic",
        name="Romantic/Tender",
        description="Loving and gentle",
        emotion=EmotionParameters(happiness=0.6, energy=0.9, tempo=0.95),
        example_instruct="Tender, affectionate tone with gentle warmth."
    ),
    EmotionPreset(
        id="mysterious",
        name="Mysterious/Dark",
        description="Enigmatic and shadowy",
        emotion=EmotionParameters(neutral=0.7, energy=0.8, tempo=0.9),
        example_instruct="Enigmatic, shadowy tone with quiet mystery."
    ),
    EmotionPreset(
        id="energetic",
        name="Energetic/Excited",
        description="High energy and fast-paced",
        emotion=EmotionParameters(happiness=0.7, surprise=0.5, energy=1.5, tempo=1.3),
        example_instruct="High-energy, rapid-fire excitement."
    ),
    EmotionPreset(
        id="calm",
        name="Calm/Serene",
        description="Peaceful and tranquil",
        emotion=EmotionParameters(neutral=0.9, energy=0.7, tempo=0.9),
        example_instruct="Serene, calm and peaceful delivery."
    ),
]


# Supported languages with their capabilities
LANGUAGE_CONFIGS = [
    TTSLanguageConfig(
        language_code="zh-CN",
        language_name="Chinese (Mandarin)",
        supported_voices=["custom", "clone", "design"],
        emotion_support=True,
        sample_rate=24000,
        model_type="neural"
    ),
    TTSLanguageConfig(
        language_code="en-US",
        language_name="English (US)",
        supported_voices=["custom", "clone", "design"],
        emotion_support=True,
        sample_rate=24000,
        model_type="neural"
    ),
    TTSLanguageConfig(
        language_code="ja-JP",
        language_name="Japanese",
        supported_voices=["custom", "clone"],
        emotion_support=True,
        sample_rate=24000,
        model_type="neural"
    ),
    TTSLanguageConfig(
        language_code="ko-KR",
        language_name="Korean",
        supported_voices=["custom"],
        emotion_support=False,
        sample_rate=24000,
        model_type="neural"
    ),
]


@router.get("/presets", response_model=ApiResponse[List[EmotionPreset]])
async def list_emotion_presets():
    """Get all available emotion presets."""
    return ApiResponse(data=EMOTION_PRESETS)


@router.get("/presets/{preset_id}", response_model=ApiResponse[EmotionPreset])
async def get_emotion_preset(preset_id: str):
    """Get a specific emotion preset."""
    for preset in EMOTION_PRESETS:
        if preset.id == preset_id:
            return ApiResponse(data=preset)

    raise NotFoundException(f"Emotion preset '{preset_id}' not found")


@router.get("/languages", response_model=ApiResponse[List[TTSLanguageConfig]])
async def list_supported_languages():
    """Get all supported languages with their capabilities."""
    return ApiResponse(data=LANGUAGE_CONFIGS)


@router.post("/generate-styled", response_model=ApiResponse[dict])
async def generate_styled_audio(
    request: VoiceStylingRequest,
    current_user: CurrentUserDep,
):
    """Generate audio with emotion and style control.

    This endpoint allows fine-grained control over the emotional delivery
    of the synthesized speech.

    Args:
        request: Text with emotion and style parameters

    Returns:
        Generated audio with metadata
    """
    # Convert emotion to TTS instruction
    instruction = _emotion_to_instruction(request.emotion)

    # Add style modifiers
    if request.style:
        style_mods = _style_to_modifiers(request.style)
        instruction = f"{instruction} {style_mods}".strip()

    # Use preset if specified
    if request.preset_id:
        for preset in EMOTION_PRESETS:
            if preset.id == request.preset_id:
                instruction = preset.example_instruct or instruction
                request.emotion = preset.emotion
                break

    # Generate audio with actual TTS engine and emotion control
    from app.services.tts_engine import TTSEngineFactory, TTSMode
    from pathlib import Path
    import numpy as np
    from pydub import AudioSegment
    import io

    tts_engine = TTSEngineFactory.create(TTSMode.LOCAL)

    try:
        # Generate base audio
        audio_data, duration = await tts_engine.generate(
            text=request.text,
            speaker=request.voice_name or "aiden",
        )

        # Apply emotion and style modifications
        audio = AudioSegment.from_file(io.BytesIO(audio_data))

        # Apply tempo
        if request.emotion.tempo and request.emotion.tempo != 1.0:
            new_frame_rate = int(audio.frame_rate * request.emotion.tempo)
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
            audio = audio.set_frame_rate(22050)

        # Apply energy (volume adjustment)
        if request.emotion.energy and request.emotion.energy != 1.0:
            db_change = 10 * np.log10(request.emotion.energy)
            audio = audio + db_change

        # Apply pitch based on emotion
        pitch_shift = 0.0
        if request.emotion.happiness and request.emotion.happiness > 0.5:
            pitch_shift = 1.5
        elif request.emotion.sadness and request.emotion.sadness > 0.5:
            pitch_shift = -2.0
        elif request.emotion.surprise and request.emotion.surprise > 0.5:
            pitch_shift = 2.0

        if pitch_shift != 0:
            new_sample_rate = int(audio.frame_rate * (2.0 ** (pitch_shift / 12.0)))
            audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
            audio = audio.set_frame_rate(22050)

        # Export modified audio
        output = io.BytesIO()
        audio.export(output, format="mp3")
        audio_data = output.read()

        # Save audio file
        audio_id = uuid.uuid4().hex[:8]
        audio_dir = Path("./static/audio/styled")
        audio_dir.mkdir(parents=True, exist_ok=True)

        audio_path = audio_dir / f"styled_{audio_id}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        return ApiResponse(
            data={
                "audio_url": f"/static/audio/styled/styled_{audio_id}.mp3",
                "duration": duration,
                "instruction": instruction,
                "emotion_applied": request.emotion.model_dump(exclude_none=True),
                "message": "Styled audio generated successfully with emotion control",
            }
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error={
                "code": "TTS_ERROR",
                "message": f"Failed to generate styled audio: {str(e)}"
            }
        )


@router.post("/convert-voice", response_model=ApiResponse[dict])
async def convert_voice(
    request: VoiceConversionRequest,
    current_user: CurrentUserDep,
):
    """Convert voice characteristics (pitch and speed adjustment).

    This is a simplified voice conversion that adjusts:
    - Pitch: Makes voice deeper or higher
    - Speed: Changes speaking rate

    For full voice-to-voice cloning, integration with RVC or GPT-SoVITS is required.

    Args:
        request: Voice conversion parameters

    Returns:
        Converted audio file info
    """
    from app.services.audio_processor import AudioProcessor
    from pathlib import Path

    audio_processor = AudioProcessor()

    # Resolve source audio path
    source_path = Path("./static") / request.source_audio_path.lstrip("/")

    if not source_path.exists():
        raise NotFoundException(f"Source audio file not found: {request.source_audio_path}")

    # Perform voice conversion (pitch/speed adjustment)
    result = await audio_processor.convert_voice(
        str(source_path),
        pitch_shift=request.pitch_shift or 0.0,
        speed_factor=request.speed_factor or 1.0,
        preserve_timing=request.preserve_timing,
    )

    if not result.get("success"):
        return ApiResponse(
            success=False,
            error={
                "code": "CONVERSION_FAILED",
                "message": result.get("error", "Voice conversion failed")
            }
        )

    return ApiResponse(
        data={
            "converted_audio_url": result["converted_audio_url"],
            "source_audio_path": request.source_audio_path,
            "target_voice_id": request.target_voice_id,
            "preserve_timing": request.preserve_timing,
            "preserve_prosody": request.preserve_prosody,
            "original_duration": result.get("original_duration"),
            "converted_duration": result.get("converted_duration"),
            "processing_steps": result.get("processing_steps", []),
            "message": "Voice conversion completed (pitch/speed adjustment)",
        }
    )


@router.post("/batch-clone", response_model=ApiResponse[dict])
async def batch_voice_clone(
    current_user: CurrentUserDep,
    voice_samples: List[UploadFile] = File(...),
    voice_name: str = Form(...),
    description: str | None = Form(None),
    language: str = Form("zh"),
    transcripts: List[str] = Form(None),
):
    """Clone a voice from multiple audio samples.

    Uses multiple audio samples to create a more robust voice clone.
    Modern systems can work with just 1-5 minutes of audio.

    Args:
        voice_samples: Multiple audio files (5-15 seconds each recommended)
        voice_name: Name for the cloned voice
        description: Optional description
        language: Language code
        transcripts: Optional transcripts for each sample

    Returns:
        Cloned voice configuration
    """
    import os
    from pathlib import Path
    from app.services.voice_cloner import get_voice_cloner

    voice_cloner = get_voice_cloner()

    # Save uploaded samples
    voice_id = uuid.uuid4().hex[:8]
    user_dir = Path("./static/uploads/voices") / current_user.id / voice_id
    user_dir.mkdir(parents=True, exist_ok=True)

    sample_paths = []
    sample_transcripts = []

    for i, sample in enumerate(voice_samples):
        if not sample.filename:
            continue

        file_path = user_dir / f"sample_{i}.wav"
        content = await sample.read()

        with open(file_path, "wb") as f:
            f.write(content)

        sample_paths.append(str(file_path))

        # Use provided transcript or default
        if transcripts and i < len(transcripts):
            sample_transcripts.append(transcripts[i])
        else:
            # Default transcript if none provided
            sample_transcripts.append(f"Sample text for voice cloning {i+1}")

    try:
        # Create voice profile with real audio processing
        profile = await voice_cloner.create_voice_profile(
            name=voice_name,
            audio_samples=list(zip(sample_paths, sample_transcripts)),
            user_id=current_user.id,
        )

        return ApiResponse(
            data={
                "voice_id": profile.profile_id,
                "voice_name": voice_name,
                "sample_count": len(sample_paths),
                "sample_paths": sample_paths,
                "language": language,
                "status": "completed",
                "voice_features": profile.voice_features,
                "reference_audio": profile.reference_audio,
                "message": f"Voice profile created successfully with {len(sample_paths)} samples",
            }
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error={
                "code": "VOICE_CLONE_FAILED",
                "message": f"Failed to create voice profile: {str(e)}"
            }
        )


@router.post("/enhance-speech", response_model=ApiResponse[dict])
async def enhance_speech(
    request: SpeechEnhancementRequest,
    current_user: CurrentUserDep,
):
    """Enhance audio quality with post-processing.

    Applies denoising, volume normalization, compression, and other
    enhancements to improve audio quality.

    Args:
        request: Enhancement parameters

    Returns:
        Enhanced audio file info
    """
    from app.services.audio_processor import AudioProcessor

    audio_processor = AudioProcessor()

    # Resolve audio path
    from pathlib import Path
    audio_path = Path("./static") / request.audio_path.lstrip("/")

    if not audio_path.exists():
        raise NotFoundException(f"Audio file not found: {request.audio_path}")

    # Apply enhancements
    result = await audio_processor.enhance_audio(
        str(audio_path),
        denoise=request.enhance_denoise,
        normalize_volume=request.enhance_volume,
        add_compression=request.add_compression,
        target_lufs=request.target_lufs,
    )

    if not result.get("success"):
        return ApiResponse(
            success=False,
            error={
                "code": "ENHANCEMENT_FAILED",
                "message": result.get("error", "Enhancement failed")
            }
        )

    return ApiResponse(
        data={
            "enhanced_audio_url": result["enhanced_audio_url"],
            "original_path": request.audio_path,
            "original_duration": result.get("original_duration"),
            "enhanced_duration": result.get("enhanced_duration"),
            "processing_steps": result.get("processing_steps", []),
            "enhancements_applied": {
                "denoise": request.enhance_denoise,
                "volume_normalize": request.enhance_volume,
                "compression": request.add_compression,
                "target_lufs": request.target_lufs,
            },
            "message": "Speech enhancement completed successfully",
        }
    )


def _emotion_to_instruction(emotion: EmotionParameters) -> str:
    """Convert emotion parameters to TTS instruction."""
    parts = []

    if emotion.happiness and emotion.happiness > 0.5:
        parts.append("joyful" if emotion.happiness > 0.7 else "happy")
    if emotion.sadness and emotion.sadness > 0.5:
        parts.append("melancholy" if emotion.sadness > 0.7 else "sad")
    if emotion.anger and emotion.anger > 0.5:
        parts.append("furious" if emotion.anger > 0.7 else "angry")
    if emotion.fear and emotion.fear > 0.5:
        parts.append("terrified" if emotion.fear > 0.7 else "fearful")
    if emotion.surprise and emotion.surprise > 0.5:
        parts.append("astonished" if emotion.surprise > 0.7 else "surprised")
    if emotion.neutral and emotion.neutral > 0.5:
        parts.append("calm" if emotion.neutral > 0.7 else "neutral")

    # Add delivery modifiers
    if emotion.energy:
        if emotion.energy > 1.3:
            parts.append("high energy")
        elif emotion.energy < 0.7:
            parts.append("low energy")

    if emotion.tempo:
        if emotion.tempo > 1.2:
            parts.append("rapid delivery")
        elif emotion.tempo < 0.8:
            parts.append("slow measured")

    return ", ".join(parts) if parts else "Neutral delivery"


def _style_to_modifiers(style: VoiceStyle) -> str:
    """Convert voice style to TTS modifiers."""
    parts = []

    if style.gender:
        parts.append(style.gender)

    if style.timbre:
        parts.append(style.timbre)

    if style.resonance:
        parts.append(style.resonance)

    if style.delivery:
        parts.append(style.delivery)

    return " ".join(parts) if parts else ""
