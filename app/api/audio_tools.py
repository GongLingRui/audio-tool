"""
Audio Tools API Routes
Provides advanced AI voice processing capabilities:
- Audio quality analysis
- Audio enhancement
- Speaker diarization
- Voice conversion
- Audio format conversion
- SSML processing
- Text segmentation
- Audio mixing
- Smart caching
- Batch processing
- Streaming TTS
- Multi-speaker consistency
- Quality assessment
- Dialect and multi-language support
- Model quantization
- RVC model management
- Enhanced speaker diarization (pyannote.audio)
- ASR speech-to-text (Whisper, Groq, Azure, Google)
"""

import asyncio
import io
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Response

from app.schemas.common import ApiResponse
from app.services.audio_quality_checker import (
    AudioQualityChecker,
    get_audio_quality_checker,
)
from app.services.audio_enhancement_service import AudioEnhancementService
from app.services.speaker_diarization_service import SpeakerDiarizationService
from app.services.voice_conversion_service import VoiceConversionService
from app.services.ssml_processor import get_ssml_processor
from app.services.intelligent_text_segmenter import get_intelligent_segmenter
from app.services.audio_mixer import get_audio_mixer
from app.services.smart_cache import get_smart_cache
from app.services.optimized_batch_processor import get_batch_processor
from app.services.streaming_tts_optimizer import get_streaming_optimizer
from app.services.multi_speaker_consistency_generator import get_multi_speaker_generator
from app.services.audio_quality_scorer import get_audio_quality_scorer
from app.services.dialect_multi_language_service import (
    get_dialect_service,
    DialectMultiLanguageService,
    LanguageCode,
)
from app.services.model_quantization_service import (
    get_quantization_service,
    ModelQuantizationService,
    QuantizationType,
)
from app.services.rvc_model_manager import (
    get_rvc_manager,
    RVCModelManager,
    ModelStatus,
)
from app.services.enhanced_speaker_diarization import (
    get_enhanced_diarization_service,
    EnhancedSpeakerDiarizationService,
    DiarizationBackend,
)
from app.services.asr_service import (
    get_asr_service,
    ASRService,
    ASRBackend,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
audio_enhancement_service = AudioEnhancementService()
speaker_diarization_service = SpeakerDiarizationService()
voice_conversion_service = VoiceConversionService()


# ==================== Audio Quality Analysis ====================

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
    temp_dir = Path(tempfile.gettempdir()) / "audio_quality_checks"
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
    temp_dir = Path(tempfile.gettempdir()) / "audio_quality_checks"
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


# ==================== Audio Enhancement ====================

@router.post("/enhance", response_model=ApiResponse[dict])
async def enhance_audio(
    file: UploadFile = File(..., description="Audio file to enhance"),
    denoise: bool = Form(True, description="Apply noise reduction"),
    normalize: bool = Form(True, description="Normalize audio levels"),
    remove_reverb: bool = Form(False, description="Reduce room reverb"),
    enhance_speech: bool = Form(True, description="Enhance speech clarity"),
    eq_preset: Optional[str] = Form(None, description="EQ preset (bass, treble, vocal, flat)"),
    output_format: str = Form("wav", description="Output format (wav, mp3, ogg, flac)"),
):
    """
    Enhance audio quality using AI-powered processing.

    Features:
    - Noise reduction using spectral subtraction
    - Speech enhancement with filtering
    - Audio normalization
    - Reverb reduction
    - EQ presets
    - Quality analysis

    Returns enhanced audio file with quality metrics.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "audio_enhancement"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        params = {
            "input_audio": str(input_path),
            "denoise": denoise,
            "normalize": normalize,
            "remove_reverb": remove_reverb,
            "enhance_speech": enhance_speech,
            "eq_preset": eq_preset,
            "output_format": output_format,
        }

        # Validate parameters
        is_valid, errors = await audio_enhancement_service.validate_input(params)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})

        # Process enhancement
        task_id = str(uuid.uuid4())
        result = await audio_enhancement_service.process(task_id, params)

        # Read output file
        output_path = Path(result["output_audio_path"])
        if output_path.exists():
            with output_path.open("rb") as f:
                audio_data = f.read()

        return ApiResponse(data={
            "success": True,
            "task_id": task_id,
            "metrics": result["metrics"],
            "processing_time": result["processing_time"],
            "output_filename": output_path.name,
            "audio_size": len(audio_data) if output_path.exists() else 0,
            "note": "Audio data available at output_audio_path",
        })

    except Exception as e:
        logger.error(f"Audio enhancement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-quality", response_model=ApiResponse[dict])
async def analyze_audio_quality(
    file: UploadFile = File(..., description="Audio file to analyze"),
):
    """
    Perform comprehensive audio quality analysis.

    Analyzes:
    - Overall quality score
    - Audio levels (dBFS)
    - Dynamic range
    - Clipping detection
    - Noise level
    - Frequency response
    - Recommendations

    Returns detailed quality report.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "audio_analysis"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        metrics = await audio_enhancement_service.analyze_quality(str(input_path))

        return ApiResponse(data={
            "success": True,
            "metrics": metrics,
        })

    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/convert-format", response_model=ApiResponse[dict])
async def convert_audio_format(
    file: UploadFile = File(..., description="Audio file to convert"),
    output_format: str = Form("wav", description="Target format (wav, mp3, ogg, flac)"),
    sample_rate: int = Form(24000, description="Target sample rate"),
    channels: int = Form(1, description="Number of channels (1=mono, 2=stereo)"),
):
    """
    Convert audio to different format.

    Supports:
    - Format conversion (wav, mp3, ogg, flac)
    - Sample rate conversion
    - Channel conversion (mono/stereo)

    Returns converted audio file info.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "audio_conversion"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        output_path = temp_dir / f"{input_path.stem}_converted.{output_format}"

        result = await audio_enhancement_service.convert_format(
            input_path=str(input_path),
            output_path=str(output_path),
            output_format=output_format,
            sample_rate=sample_rate,
            channels=channels,
        )

        if not result.get("success"):
            raise Exception(result.get("error", "Conversion failed"))

        return ApiResponse(data={
            "success": True,
            "output_path": result["output_path"],
            "duration": result["duration"],
            "sample_rate": result["sample_rate"],
            "channels": result["channels"],
        })

    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Speaker Diarization ====================

@router.post("/diarize", response_model=ApiResponse[dict])
async def diarize_speakers(
    file: UploadFile = File(..., description="Audio file to process"),
    min_speakers: int = Form(1, description="Minimum number of speakers"),
    max_speakers: int = Form(5, description="Maximum number of speakers"),
    language: str = Form("zh", description="Language code"),
):
    """
    Identify and separate speakers in audio.

    Features:
    - Automatic speaker detection
    - Speaker segmentation
    - Speaker count estimation
    - Timestamp extraction

    Returns speaker segments with timestamps.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "speaker_diarization"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        params = {
            "input_audio": str(input_path),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "language": language,
        }

        # Validate parameters
        is_valid, errors = await speaker_diarization_service.validate_input(params)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})

        # Process diarization
        task_id = str(uuid.uuid4())
        result = await speaker_diarization_service.process(task_id, params)

        return ApiResponse(data={
            "success": True,
            "task_id": task_id,
            "num_speakers": result["num_speakers"],
            "speakers": result["speakers"],
            "segments": result["segments"],
            "duration": result["duration"],
            "processing_time": result["processing_time"],
        })

    except Exception as e:
        logger.error(f"Speaker diarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-speaker", response_model=ApiResponse[dict])
async def extract_speaker_audio(
    file: UploadFile = File(..., description="Audio file to process"),
    speaker_id: str = Form(..., description="Speaker ID to extract (e.g., SPEAKER_00)"),
):
    """
    Extract audio segments for a specific speaker.

    Returns a new audio file containing only the specified speaker's segments.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "speaker_extraction"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        output_path = await speaker_diarization_service.extract_speaker_audio(
            audio_path=str(input_path),
            speaker_id=speaker_id,
        )

        return ApiResponse(data={
            "success": True,
            "speaker_id": speaker_id,
            "output_path": output_path,
        })

    except Exception as e:
        logger.error(f"Speaker extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-speakers", response_model=ApiResponse[dict])
async def compare_speaker_voices(
    file1: UploadFile = File(..., description="First audio file"),
    file2: UploadFile = File(..., description="Second audio file"),
):
    """
    Compare two audio samples to determine if they're from the same speaker.

    Returns similarity score and match result.
    """
    # Save uploaded files
    temp_dir = Path(tempfile.gettempdir()) / "speaker_comparison"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path1 = temp_dir / file1.filename
    with input_path1.open("wb") as f:
        content = await file1.read()
        f.write(content)

    input_path2 = temp_dir / file2.filename
    with input_path2.open("wb") as f:
        content = await file2.read()
        f.write(content)

    try:
        result = await speaker_diarization_service.compare_speakers(
            audio_path1=str(input_path1),
            audio_path2=str(input_path2),
        )

        return ApiResponse(data={
            "success": True,
            "similarity": result["similarity"],
            "same_speaker": result["same_speaker"],
            "confidence": result.get("confidence", 0.0),
        })

    except Exception as e:
        logger.error(f"Speaker comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Voice Conversion ====================

@router.post("/voice-convert", response_model=ApiResponse[dict])
async def convert_voice(
    file: UploadFile = File(..., description="Audio file to convert"),
    target_voice: str = Form(..., description="Target voice preset (original, deeper, higher, robotic, echo, telephone)"),
    similarity: float = Form(0.85, description="Similarity threshold (0.0 - 1.0)"),
    pitch_shift: int = Form(0, description="Pitch shift in semitones (-12 to +12)"),
    formant_shift: float = Form(0.0, description="Formant shift (-1.0 to 1.0)"),
):
    """
    Convert voice to different characteristics.

    Voice Presets:
    - original: Keep original voice
    - deeper: Lower pitch for deeper voice
    - higher: Higher pitch voice
    - robotic: Electronic/robotic effect
    - echo: Add echo/delay effect
    - telephone: Telephone quality effect

    Returns converted audio with quality metrics.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "voice_conversion"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        params = {
            "source_audio": str(input_path),
            "target_voice": target_voice,
            "similarity": similarity,
            "pitch_shift": pitch_shift,
            "formant_shift": formant_shift,
        }

        # Validate parameters
        is_valid, errors = await voice_conversion_service.validate_input(params)
        if not is_valid:
            raise HTTPException(status_code=400, detail={"errors": errors})

        # Process conversion
        task_id = str(uuid.uuid4())
        result = await voice_conversion_service.process(task_id, params)

        return ApiResponse(data={
            "success": True,
            "task_id": task_id,
            "output_audio_path": result["output_audio_path"],
            "quality_score": result["quality_score"],
            "processing_time": result["processing_time"],
            "target_voice": result["target_voice"],
        })

    except Exception as e:
        logger.error(f"Voice conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/voice-presets", response_model=ApiResponse[dict])
async def get_voice_presets():
    """
    Get available voice conversion presets.

    Returns list of available voice configurations.
    """
    presets = await voice_conversion_service.get_available_voices()

    return ApiResponse(data={
        "presets": presets,
        "count": len(presets),
    })


@router.post("/voice-profile", response_model=ApiResponse[dict])
async def create_voice_profile(
    file: UploadFile = File(..., description="Reference audio sample"),
    profile_name: str = Form(..., description="Name for the voice profile"),
):
    """
    Create a voice profile from audio sample.

    Extracts voice characteristics for use in voice conversion.
    """
    # Save uploaded file
    temp_dir = Path(tempfile.gettempdir()) / "voice_profiles"
    temp_dir.mkdir(parents=True, exist_ok=True)

    input_path = temp_dir / file.filename
    with input_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        profile = await voice_conversion_service.voice_profile_to_json(
            audio_path=str(input_path),
            profile_name=profile_name,
        )

        return ApiResponse(data={
            "success": True,
            "profile": profile,
        })

    except Exception as e:
        logger.error(f"Voice profile creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System Info ====================

@router.get("/capabilities", response_model=ApiResponse[dict])
async def get_capabilities():
    """
    Get available audio processing capabilities.

    Returns information about supported features and formats.
    """
    return ApiResponse(data={
        "audio_enhancement": {
            "noise_reduction": True,
            "normalization": True,
            "eq_presets": ["bass", "treble", "vocal", "flat"],
            "formats": ["wav", "mp3", "ogg", "flac"],
        },
        "speaker_diarization": {
            "auto_detection": True,
            "speaker_extraction": True,
            "speaker_comparison": True,
            "max_speakers": 10,
        },
        "voice_conversion": {
            "presets": ["original", "deeper", "higher", "robotic", "echo", "telephone"],
            "pitch_shift": True,
            "formant_shift": True,
        },
        "quality_analysis": {
            "level_analysis": True,
            "clipping_detection": True,
            "dynamic_range": True,
            "recommendations": True,
        },
        "ssml_processing": {
            "w3c_ssml_11": True,
            "special_text": True,
            "chinese_support": True,
        },
        "streaming_tts": {
            "first_byte_latency_ms": 150,
            "parallel_generation": True,
        },
        "caching": {
            "l1_memory": True,
            "l2_redis": True,
            "l3_file": True,
        },
    })


# ==================== SSML Processing ====================

@router.post("/ssml/process", response_model=ApiResponse[dict])
async def process_ssml(
    ssml: str = Form(..., description="SSML text to process"),
    voice_config: str = Form("{}", description="Voice configuration as JSON string"),
):
    """
    Process SSML text according to W3C SSML 1.1 standard.

    Supports:
    - All W3C SSML 1.1 elements
    - Special text formatting (dates, times, numbers, currency)
    - Chinese number conversion
    - Emotion and prosody control

    Returns processed plain text ready for TTS.
    """
    import json

    try:
        processor = get_ssml_processor()
        voice_cfg = json.loads(voice_config) if voice_config else {}

        result = await processor.process(ssml, voice_cfg)

        return ApiResponse(data={
            "success": True,
            "processed_text": result.processed_text,
            "special_texts": result.special_texts,
            "metadata": result.metadata,
        })

    except Exception as e:
        logger.error(f"SSML processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ssml/special-format", response_model=ApiResponse[dict])
async def format_special_text(
    text: str = Form(..., description="Text to format"),
    format_type: str = Form(..., description="Format type (date, time, number, currency, telephone)"),
):
    """
    Format special text types for natural TTS pronunciation.

    Format types:
    - date: Format dates (e.g., 2025-02-18 -> 2025年2月18日)
    - time: Format times (e.g., 14:30 -> 下午两点半)
    - number: Format numbers (e.g., 12345 -> 一万二千三百四十五)
    - currency: Format currency (e.g., 10000 -> 1.0万元)
    - telephone: Format telephone numbers digit by digit

    Returns formatted text.
    """
    try:
        processor = get_ssml_processor()

        if format_type == "date":
            result = processor._format_date(text, "ymd")
        elif format_type == "time":
            result = processor._format_time(text, "hms12")
        elif format_type == "number":
            result = processor._format_number(text, "cardinal")
        elif format_type == "currency":
            result = processor._format_currency(text, "CNY")
        elif format_type == "telephone":
            result = processor._format_telephone(text)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown format type: {format_type}")

        return ApiResponse(data={
            "success": True,
            "original": text,
            "formatted": result,
            "format_type": format_type,
        })

    except Exception as e:
        logger.error(f"Special text formatting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Text Segmentation ====================

@router.post("/text/segment", response_model=ApiResponse[dict])
async def segment_text(
    text: str = Form(..., description="Text to segment"),
    first_chunk_size: int = Form(30, description="Size of first chunk in characters"),
    subsequent_chunk_size: int = Form(80, description="Size of subsequent chunks"),
    for_streaming: bool = Form(True, description="Optimize for streaming"),
):
    """
    Intelligently segment text for TTS processing.

    Features:
    - Semantic-aware segmentation
    - Dialogue detection and attribution
    - Natural pause calculation
    - Streaming optimization

    Returns list of text segments with metadata.
    """
    try:
        segmenter = get_intelligent_segmenter()

        if for_streaming:
            segments = await segmenter.segment_for_streaming(
                text, first_chunk_size, subsequent_chunk_size
            )
            result_data = [
                {
                    "text": s["text"],
                    "pause_after": s.get("pause_after", 0.0),
                    "metadata": s.get("metadata", {}),
                }
                for s in segments
            ]
        else:
            segments = await segmenter.segment_for_batch(text)
            result_data = [
                {
                    "text": s["text"],
                    "is_dialogue": s.get("is_dialogue", False),
                    "speaker": s.get("speaker"),
                    "emotion": s.get("emotion"),
                }
                for s in segments
            ]

        return ApiResponse(data={
            "success": True,
            "segments": result_data,
            "count": len(result_data),
        })

    except Exception as e:
        logger.error(f"Text segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text/detect-dialogue", response_model=ApiResponse[dict])
async def detect_dialogue(
    text: str = Form(..., description="Text to analyze for dialogue"),
):
    """
    Detect dialogue patterns and speakers in text.

    Identifies:
    - Dialogue segments
    - Speaker markers
    - Emotion indicators
    - Action descriptions

    Returns dialogue structure with speaker attribution.
    """
    try:
        segmenter = get_intelligent_segmenter()
        result = await segmenter.detect_dialogue(text)

        return ApiResponse(data={
            "success": True,
            "has_dialogue": result["has_dialogue"],
            "speakers": result["speakers"],
            "segments": result["segments"],
        })

    except Exception as e:
        logger.error(f"Dialogue detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Audio Mixing ====================

@router.post("/audio/mix", response_model=ApiResponse[dict])
async def mix_audio(
    speech_audio: UploadFile = File(..., description="Main speech audio"),
    background_music: UploadFile = File(None, description="Background music file (optional)"),
    music_volume: float = Form(0.2, description="Music volume (0.0 - 1.0)"),
    ducking: bool = Form(True, description="Apply ducking (lower music during speech)"),
    ducking_amount: float = Form(6.0, description="Ducking amount in dB"),
    crossfade: float = Form(0.0, description="Crossfade duration in seconds"),
    normalize: bool = Form(True, description="Normalize output"),
):
    """
    Mix speech audio with background music and effects.

    Features:
    - Background music mixing with looping
    - Automatic ducking (lowers music during speech)
    - Crossfading between segments
    - Volume normalization
    - Multiple audio format support

    Returns mixed audio bytes.
    """
    try:
        mixer = get_audio_mixer()

        # Read speech audio
        speech_data = await speech_audio.read()

        # Read background music if provided
        music_data = None
        if background_music:
            music_data = await background_music.read()

        # Mix audio
        result = await mixer.mix_audio(
            speech_audio=speech_data,
            background_music=music_data,
            music_volume=music_volume,
            ducking=ducking,
            ducking_amount=ducking_amount,
            crossfade=crossfade,
            normalize=normalize,
        )

        return ApiResponse(data={
            "success": True,
            "audio_size": len(result),
            "note": "Mixed audio returned in response",
        })

    except Exception as e:
        logger.error(f"Audio mixing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/crossfade", response_model=ApiResponse[dict])
async def create_crossfade(
    audio1: UploadFile = File(..., description="First audio segment"),
    audio2: UploadFile = File(..., description="Second audio segment"),
    crossfade_duration: float = Form(1.0, description="Crossfade duration in seconds"),
):
    """
    Create crossfade between two audio segments.

    Returns crossfaded audio bytes.
    """
    try:
        mixer = get_audio_mixer()

        audio1_data = await audio1.read()
        audio2_data = await audio2.read()

        result = await mixer.create_audio_crossfade(
            audio1_data, audio2_data, crossfade_duration
        )

        return ApiResponse(data={
            "success": True,
            "audio_size": len(result),
        })

    except Exception as e:
        logger.error(f"Crossfade error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/splice", response_model=ApiResponse[dict])
async def splice_audio(
    files: List[UploadFile] = File(..., description="Audio segments to splice"),
    crossfade: float = Form(0.5, description="Crossfade duration in seconds"),
):
    """
    Splice multiple audio segments together with crossfade.

    Returns spliced audio bytes.
    """
    try:
        mixer = get_audio_mixer()

        segments = []
        for file in files:
            data = await file.read()
            segments.append(data)

        result = await mixer.splice_audio(segments, crossfade)

        return ApiResponse(data={
            "success": True,
            "audio_size": len(result),
            "segments_processed": len(segments),
        })

    except Exception as e:
        logger.error(f"Audio splicing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Smart Caching ====================

@router.get("/cache/stats", response_model=ApiResponse[dict])
async def get_cache_stats():
    """
    Get cache statistics across all levels.

    Returns:
    - Memory cache stats
    - Redis cache stats (if enabled)
    - Hit rates and usage
    """
    try:
        cache_manager = get_smart_cache()
        stats = cache_manager.get_stats()

        return ApiResponse(data=stats)

    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear", response_model=ApiResponse[dict])
async def clear_cache():
    """
    Clear all cache levels.

    Clears memory and Redis cache.
    """
    try:
        cache_manager = get_smart_cache()
        await cache_manager.clear()

        return ApiResponse(data={
            "success": True,
            "message": "All cache levels cleared",
        })

    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Batch Processing ====================

@router.post("/batch/process", response_model=ApiResponse[dict])
async def process_batch(
    items: str = Form(..., description="Batch items as JSON string"),
    show_progress: bool = Form(False, description="Show progress during processing"),
):
    """
    Process a batch of TTS items with intelligent batching.

    Features:
    - Intelligent batching by text length
    - Parallel processing with configurable workers
    - Result caching
    - Automatic retry on failure

    Args:
        items: JSON string with list of items, each containing:
            - text: Text to synthesize
            - voice_config: Voice configuration
            - emotion: Optional emotion parameters

    Returns batch processing results with metrics.
    """
    import json

    try:
        processor = get_batch_processor()
        items_list = json.loads(items)

        # Mock generator function (in real use, would be TTS generator)
        async def mock_generator(text, voice_config, emotion):
            # Placeholder - in production, use actual TTS
            return b"mock_audio_data"

        result = await processor.process_batch(
            items_list, mock_generator, show_progress
        )

        return ApiResponse(data={
            "success": True,
            "total_duration": result.total_duration,
            "processing_time": result.processing_time,
            "success_count": result.success_count,
            "failure_count": result.failure_count,
            "items_per_second": result.items_per_second,
            "items": [
                {
                    "id": item.id,
                    "text": item.text,
                    "has_error": item.error is not None,
                    "error": item.error,
                }
                for item in result.items
            ],
        })

    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Streaming TTS ====================

@router.post("/stream/generate", response_model=ApiResponse[dict])
async def generate_streaming(
    text: str = Form(..., description="Text to synthesize"),
    voice_config: str = Form("{}", description="Voice configuration as JSON"),
    first_chunk_size: int = Form(30, description="First chunk size"),
    subsequent_chunk_size: int = Form(80, description="Subsequent chunk size"),
):
    """
    Generate streaming audio with ultra-low first-byte latency.

    Target: <150ms first-byte latency

    Features:
    - First chunk optimization (small, fast)
    - Parallel chunk generation
    - Incremental streaming
    - Latency monitoring

    Returns streaming chunks with metadata.
    """
    import json

    try:
        optimizer = get_streaming_optimizer(
            first_chunk_size=first_chunk_size,
            subsequent_chunk_size=subsequent_chunk_size,
        )

        voice_cfg = json.loads(voice_config) if voice_config else {}

        # Mock generator (in production, use actual TTS)
        async def mock_generator(text, voice_config, emotion):
            await asyncio.sleep(0.1)  # Simulate TTS latency
            return b"mock_audio_chunk"

        chunks = []
        async for chunk in optimizer.generate_streaming(
            text, mock_generator, voice_cfg
        ):
            chunks.append({
                "index": chunk.index,
                "text": chunk.text,
                "is_first": chunk.is_first,
                "is_final": chunk.is_final,
                "latency_ms": chunk.latency_ms,
            })

        return ApiResponse(data={
            "success": True,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "first_byte_latency": chunks[0]["latency_ms"] if chunks else 0,
        })

    except Exception as e:
        logger.error(f"Streaming generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stream/measure-latency", response_model=ApiResponse[dict])
async def measure_streaming_latency(
    text: str = Form("这是一段测试文本，用于测量流式TTS的延迟。", description="Test text"),
    trials: int = Form(5, description="Number of trials"),
):
    """
    Measure actual streaming TTS latency.

    Returns latency metrics including:
    - First-byte latency (avg, min, max)
    - Total stream latency
    - Target comparison
    """
    try:
        optimizer = get_streaming_optimizer()

        # Mock generator
        async def mock_generator(text, voice_config, emotion):
            await asyncio.sleep(0.1)
            return b"mock_audio"

        voice_config = {}
        metrics = await optimizer.measure_latency(
            text, mock_generator, voice_config, trials
        )

        return ApiResponse(data=metrics)

    except Exception as e:
        logger.error(f"Latency measurement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Multi-Speaker Consistency ====================

@router.post("/speakers/create-profile", response_model=ApiResponse[dict])
async def create_speaker_profile(
    speaker_id: str = Form(..., description="Unique speaker identifier"),
    name: str = Form(..., description="Speaker display name"),
    reference_audio: UploadFile = File(None, description="Reference audio sample"),
    voice_config: str = Form("{}", description="Voice configuration as JSON"),
):
    """
    Create a new speaker profile for consistency tracking.

    Returns speaker profile with extracted characteristics.
    """
    import json

    try:
        generator = get_multi_speaker_generator()

        reference_data = None
        if reference_audio:
            reference_data = await reference_audio.read()

        voice_cfg = json.loads(voice_config) if voice_config else {}

        profile = await generator.create_speaker_profile(
            speaker_id, name, reference_data, voice_cfg
        )

        return ApiResponse(data={
            "success": True,
            "speaker_id": profile.speaker_id,
            "name": profile.name,
            "has_reference": profile.reference_audio is not None,
            "characteristics": profile.characteristics,
        })

    except Exception as e:
        logger.error(f"Profile creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/speakers/list", response_model=ApiResponse[dict])
async def list_speakers():
    """
    List all available speaker profiles.

    Returns speaker profiles with basic info.
    """
    try:
        generator = get_multi_speaker_generator()
        speakers = generator.list_speakers()

        return ApiResponse(data={
            "speakers": speakers,
            "count": len(speakers),
        })

    except Exception as e:
        logger.error(f"List speakers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speakers/check-consistency", response_model=ApiResponse[dict])
async def check_speaker_consistency(
    speaker_id: str = Form(..., description="Speaker ID to check against"),
    audio: UploadFile = File(..., description="Audio to check"),
):
    """
    Check if audio matches speaker profile.

    Returns consistency score and similarity metrics.
    """
    try:
        generator = get_multi_speaker_generator()
        audio_data = await audio.read()

        is_consistent, similarity = await generator.check_consistency(
            speaker_id, audio_data
        )

        return ApiResponse(data={
            "speaker_id": speaker_id,
            "is_consistent": is_consistent,
            "similarity": similarity,
            "threshold": generator.consistency_threshold,
        })

    except Exception as e:
        logger.error(f"Consistency check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/speakers/compare", response_model=ApiResponse[dict])
async def compare_speakers(
    speaker_id1: str = Form(..., description="First speaker ID"),
    speaker_id2: str = Form(..., description="Second speaker ID"),
):
    """
    Compare two speaker profiles.

    Returns similarity metrics between speakers.
    """
    try:
        generator = get_multi_speaker_generator()
        comparison = await generator.compare_speakers(speaker_id1, speaker_id2)

        return ApiResponse(data=comparison)

    except Exception as e:
        logger.error(f"Speaker comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Quality Assessment ====================

@router.post("/quality/assess", response_model=ApiResponse[dict])
async def assess_quality(
    file: UploadFile = File(..., description="Audio to assess"),
    reference: UploadFile = File(None, description="Reference audio for comparison"),
):
    """
    Comprehensively assess audio quality.

    Features:
    - Speech clarity analysis
    - Naturalness assessment
    - Voice consistency scoring
    - Dynamic range evaluation
    - Signal-to-noise ratio
    - Artifact detection

    Returns comprehensive quality report with recommendations.
    """
    import tempfile
    import os

    try:
        scorer = get_audio_quality_scorer()

        # Save uploaded file
        temp_dir = Path(tempfile.gettempdir()) / "quality_assessment"
        temp_dir.mkdir(parents=True, exist_ok=True)

        temp_path = temp_dir / file.filename
        with temp_path.open("wb") as f:
            content = await file.read()
            f.write(content)

        # Save reference if provided
        ref_path = None
        if reference:
            ref_path = temp_dir / f"ref_{reference.filename}"
            with ref_path.open("wb") as f:
                content = await reference.read()
                f.write(content)

        # Score audio
        metrics = await scorer.score_audio(str(temp_path), str(ref_path) if ref_path else None)

        # Get quality report
        report = scorer.get_quality_report(metrics)

        # Clean up
        temp_path.unlink()
        if ref_path:
            ref_path.unlink()

        return ApiResponse(data=report)

    except Exception as e:
        logger.error(f"Quality assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quality/batch", response_model=ApiResponse[dict])
async def assess_batch_quality(
    files: List[UploadFile] = File(..., description="Multiple audio files"),
):
    """
    Assess quality of multiple audio files.

    Returns individual and combined quality reports.
    """
    import tempfile

    try:
        scorer = get_audio_quality_scorer()

        # Save uploaded files
        temp_dir = Path(tempfile.gettempdir()) / "batch_quality_assessment"
        temp_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        temp_paths = []
        for file in files:
            temp_path = temp_dir / file.filename
            temp_paths.append(temp_path)
            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)
            paths.append(str(temp_path))

        try:
            # Batch score
            metrics_list = await scorer.batch_score(paths)

            # Generate reports
            reports = [scorer.get_quality_report(m) for m in metrics_list]

            # Calculate average
            avg_score = sum(m.overall_score for m in metrics_list) / len(metrics_list)

            return ApiResponse(data={
                "total_files": len(reports),
                "average_score": avg_score,
                "reports": reports,
            })

        finally:
            # Clean up
            for temp_path in temp_paths:
                if temp_path.exists():
                    temp_path.unlink()

    except Exception as e:
        logger.error(f"Batch quality assessment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quality/thresholds", response_model=ApiResponse[dict])
async def get_quality_thresholds():
    """
    Get quality thresholds for assessment.

    Returns thresholds for excellent, good, fair, and poor quality.
    """
    try:
        scorer = get_audio_quality_scorer()
        thresholds = scorer._quality_thresholds

        return ApiResponse(data={
            "thresholds": thresholds,
            "description": {
                "excellent": "90+ - Excellent quality",
                "good": "75+ - Good quality",
                "fair": "60+ - Fair quality",
                "poor": "<60 - Poor quality",
            },
        })

    except Exception as e:
        logger.error(f"Get thresholds error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Dialect and Multi-Language Support ====================

@router.post("/dialect/detect-language", response_model=ApiResponse[dict])
async def detect_language(
    text: str = Form(..., description="Text to analyze"),
):
    """
    Detect language and dialect in text.

    Supports:
    - Mandarin Chinese (zh-CN)
    - Cantonese (zh-HK)
    - Hakka (zh-HAK)
    - Min Nan (zh-MIN)
    - Wu/Shanghainese (zh-WU)
    - English (en-US)
    - Japanese (ja-JP)
    - Korean (ko-KR)

    Returns detected language segments with confidence scores.
    """
    try:
        dialect_service = get_dialect_service()
        segments = await dialect_service.detect_language(text)

        return ApiResponse(data={
            "text": text,
            "segments": [
                {
                    "text": s.text,
                    "language": s.language.value,
                    "confidence": s.confidence,
                    "position": (s.start_pos, s.end_pos),
                }
                for s in segments
            ],
        })

    except Exception as e:
        logger.error(f"Language detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dialect/convert", response_model=ApiResponse[dict])
async def convert_dialect_text(
    text: str = Form(..., description="Original text"),
    target_dialect: str = Form(..., description="Target dialect code"),
):
    """
    Convert text to dialect-specific characters.

    Example: "这里什么" -> "呢度乜嘢" (Cantonese)
    """
    try:
        dialect_service = get_dialect_service()

        # Parse dialect code
        try:
            target_lang = LanguageCode(target_dialect)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dialect code: {target_dialect}"
            )

        converted_text = await dialect_service.convert_dialect_text(text, target_lang)

        return ApiResponse(data={
            "original_text": text,
            "converted_text": converted_text,
            "target_dialect": target_dialect,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dialect conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dialect/process-mixed", response_model=ApiResponse[dict])
async def process_mixed_language(
    text: str = Form(..., description="Mixed-language text"),
):
    """
    Process mixed-language text with appropriate voice configurations.

    Automatically detects different languages in the text and provides
    appropriate TTS configurations for each segment.
    """
    try:
        dialect_service = get_dialect_service()
        result = await dialect_service.process_mixed_language(text)

        return ApiResponse(data={
            "original_text": text,
            "segments": result,
        })

    except Exception as e:
        logger.error(f"Mixed language processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dialect/supported", response_model=ApiResponse[dict])
async def get_supported_dialects():
    """
    Get list of all supported dialects and languages.
    """
    try:
        dialect_service = get_dialect_service()
        dialects = await dialect_service.get_supported_dialects()

        return ApiResponse(data={
            "dialects": dialects,
        })

    except Exception as e:
        logger.error(f"Get supported dialects error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dialect/validate", response_model=ApiResponse[dict])
async def validate_dialect(
    dialect: str = Form(..., description="Dialect code to validate"),
):
    """
    Validate if a dialect is supported and get availability information.
    """
    try:
        dialect_service = get_dialect_service()
        try:
            lang_code = LanguageCode(dialect)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dialect code: {dialect}"
            )

        support_info = await dialect_service.validate_dialect_support(lang_code)

        return ApiResponse(data=support_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dialect validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dialect/suggest", response_model=ApiResponse[dict])
async def suggest_dialect(
    text: str = Form(..., description="Content text"),
    target_audience: str = Form("mainland", description="Target audience: mainland, hk, taiwan, overseas"),
):
    """
    Suggest appropriate dialect/language for content based on text and target audience.
    """
    try:
        dialect_service = get_dialect_service()
        suggestion = await dialect_service.suggest_dialect_for_content(text, target_audience)

        return ApiResponse(data=suggestion)

    except Exception as e:
        logger.error(f"Dialect suggestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Model Quantization ====================

@router.post("/quantization/quantize", response_model=ApiResponse[dict])
async def quantize_model(
    model_file: UploadFile = File(..., description="Model file to quantize"),
    quantization_type: str = Form("int8", description="Quantization type: int8, fp16, dynamic, gptq, awq"),
    model_format: str = Form("pytorch", description="Model format: pytorch, onnx, transformers"),
    calibration_texts: str = Form(None, description="Comma-separated calibration texts (for static quantization)"),
):
    """
    Quantize a TTS model for faster inference.

    Benefits:
    - INT8: ~2.5x faster, 75% size reduction
    - FP16: ~1.5x faster, 50% size reduction
    - Dynamic: ~2x faster, automatic calibration
    """
    try:
        # Save uploaded model
        temp_dir = Path(tempfile.gettempdir()) / "model_quantization"
        temp_dir.mkdir(parents=True, exist_ok=True)

        model_path = temp_dir / model_file.filename
        with model_path.open("wb") as f:
            content = await model_file.read()
            f.write(content)

        try:
            # Parse quantization type
            try:
                quant_type = QuantizationType(quantization_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid quantization type: {quantization_type}"
                )

            # Parse calibration data
            calibration_data = None
            if calibration_texts:
                calibration_data = [t.strip() for t in calibration_texts.split(",")]

            # Perform quantization
            quant_service = get_quantization_service()
            result = await quant_service.quantize_model(
                str(model_path),
                quant_type,
                model_format,
                calibration_data,
            )

            return ApiResponse(data={
                "success": result.success,
                "quantization_type": result.quantization_type.value,
                "original_size_mb": round(result.original_size_mb, 2),
                "quantized_size_mb": round(result.quantized_size_mb, 2),
                "compression_ratio": round(result.compression_ratio, 2),
                "inference_speedup": round(result.inference_speedup, 2),
                "calibration_time": round(result.calibration_time, 2),
                "quantized_path": result.model_path,
                "error": result.error,
            })

        finally:
            # Clean up original model
            if model_path.exists():
                model_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model quantization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantization/compare", response_model=ApiResponse[dict])
async def compare_quantized_models(
    original_path: str = Form(..., description="Path to original model"),
    quantized_path: str = Form(..., description="Path to quantized model"),
    test_texts: str = Form("测试文本", description="Comma-separated test texts"),
):
    """
    Compare original and quantized models.

    Returns size reduction, latency comparison, and quality metrics.
    """
    try:
        quant_service = get_quantization_service()

        # Parse test texts
        texts = [t.strip() for t in test_texts.split(",")]

        result = await quant_service.compare_models(original_path, quantized_path, texts)

        return ApiResponse(data=result)

    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quantization/models", response_model=ApiResponse[dict])
async def list_quantized_models():
    """
    List all quantized models.
    """
    try:
        quant_service = get_quantization_service()
        models = await quant_service.list_quantized_models()

        return ApiResponse(data={
            "models": models,
            "total": len(models),
        })

    except Exception as e:
        logger.error(f"List quantized models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantization/validate", response_model=ApiResponse[dict])
async def validate_quantized_model(
    model_path: str = Form(..., description="Path to quantized model"),
):
    """
    Validate a quantized model.

    Checks file existence, readability, and metadata integrity.
    """
    try:
        quant_service = get_quantization_service()
        result = await quant_service.validate_quantized_model(model_path)

        return ApiResponse(data=result)

    except Exception as e:
        logger.error(f"Validate quantized model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quantization/cleanup", response_model=ApiResponse[dict])
async def cleanup_quantized_models(
    older_than_days: int = Form(30, description="Delete models older than this many days"),
):
    """
    Clean up old quantized models.

    Useful for managing disk space.
    """
    try:
        quant_service = get_quantization_service()
        await quant_service.cleanup_quantized_models(older_than_days)

        return ApiResponse(data={
            "message": f"Cleaned up quantized models older than {older_than_days} days",
        })

    except Exception as e:
        logger.error(f"Cleanup quantized models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RVC Model Management ====================

@router.get("/rvc/models", response_model=ApiResponse[dict])
async def list_rvc_models(
    language: Optional[str] = None,
    gender: Optional[str] = None,
    status: Optional[str] = None,
):
    """
    List all RVC models with optional filtering.

    Filters:
    - language: Filter by language (e.g., "zh-CN", "en-US")
    - gender: Filter by gender ("male", "female")
    - status: Filter by status ("available", "training", "error", "incomplete")
    """
    try:
        rvc_manager = get_rvc_manager()

        # Parse status if provided
        status_filter = None
        if status:
            try:
                status_filter = ModelStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}"
                )

        models = rvc_manager.list_models(
            language=language,
            gender=gender,
            status=status_filter,
        )

        return ApiResponse(data={
            "models": [m.to_dict() for m in models],
            "total": len(models),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List RVC models error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rvc/models/{model_id}", response_model=ApiResponse[dict])
async def get_rvc_model(model_id: str):
    """
    Get details of a specific RVC model.
    """
    try:
        rvc_manager = get_rvc_manager()
        model = rvc_manager.get_model(model_id)

        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        return ApiResponse(data=model.to_dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get RVC model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rvc/models/upload", response_model=ApiResponse[dict])
async def upload_rvc_model(
    model_file: UploadFile = File(..., description="RVC model file (.pth)"),
    model_id: str = Form(..., description="Unique model identifier"),
    name: str = Form(..., description="Model display name"),
    description: str = Form("", description="Model description"),
    index_file: UploadFile = File(None, description="Index file (.index)"),
    language: str = Form("zh-CN", description="Model language"),
    gender: str = Form("female", description="Model gender"),
    sample_rate: int = Form(48000, description="Sample rate"),
    f0_method: str = Form("rmvpe", description="F0 extraction method"),
):
    """
    Upload and register a new RVC model.

    RVC models are used for voice conversion.
    """
    try:
        rvc_manager = get_rvc_manager()

        # Read files
        model_bytes = await model_file.read()
        index_bytes = await index_file.read() if index_file else None

        # Register model
        model = await rvc_manager.upload_model(
            model_file=model_bytes,
            model_id=model_id,
            name=name,
            description=description,
            index_file=index_bytes,
            language=language,
            gender=gender,
            sample_rate=sample_rate,
            f0_method=f0_method,
        )

        return ApiResponse(data={
            "message": f"Model {model_id} uploaded successfully",
            "model": model.to_dict(),
        })

    except Exception as e:
        logger.error(f"Upload RVC model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rvc/models/{model_id}", response_model=ApiResponse[dict])
async def delete_rvc_model(model_id: str):
    """
    Delete an RVC model.
    """
    try:
        rvc_manager = get_rvc_manager()
        success = await rvc_manager.delete_model(model_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        return ApiResponse(data={
            "message": f"Model {model_id} deleted successfully",
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete RVC model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rvc/convert")
async def convert_audio_rvc(
    audio_file: UploadFile = File(..., description="Source audio file"),
    model_id: str = Form(..., description="RVC model to use"),
    pitch_shift: int = Form(0, description="Pitch shift in semitones"),
    f0_method: str = Form("rmvpe", description="F0 extraction method"),
    filter_radius: int = Form(3, description="Filter radius"),
    rms_mix_rate: float = Form(0.25, description="RMS mix rate"),
    protect: float = Form(0.33, description="Consonant protection"),
):
    """
    Convert audio using RVC model.

    Performs voice conversion using the specified RVC model.
    """
    try:
        rvc_manager = get_rvc_manager()

        # Read audio
        source_audio = await audio_file.read()

        # Convert
        converted_audio = await rvc_manager.convert_audio(
            source_audio=source_audio,
            model_id=model_id,
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        # Return converted audio
        return Response(
            content=converted_audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=converted_{model_id}.wav"
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"RVC conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rvc/models/{model_id}/validate", response_model=ApiResponse[dict])
async def validate_rvc_model(model_id: str):
    """
    Validate an RVC model.

    Checks model file, index file, parameters, and file size.
    """
    try:
        rvc_manager = get_rvc_manager()
        result = await rvc_manager.validate_model(model_id)

        return ApiResponse(data=result)

    except Exception as e:
        logger.error(f"Validate RVC model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rvc/models/{model_id}/train", response_model=ApiResponse[dict])
async def train_rvc_model(
    model_id: str,
    training_files: List[UploadFile] = File(..., description="Audio samples for training"),
    epochs: int = Form(100, description="Number of training epochs"),
    batch_size: int = Form(8, description="Batch size"),
    rvc_version: str = Form("v2", description="RVC version (v1 or v2)"),
):
    """
    Train a new RVC model (framework).

    Note: Actual RVC training requires external RVC training pipeline.
    This endpoint prepares the training data and configuration.
    """
    try:
        rvc_manager = get_rvc_manager()

        # Read training samples
        training_data = [await f.read() for f in training_files]

        # Prepare training
        result = await rvc_manager.train_model(
            model_id=model_id,
            training_data=training_data,
            epochs=epochs,
            batch_size=batch_size,
            rvc_version=rvc_version,
        )

        return ApiResponse(data=result)

    except Exception as e:
        logger.error(f"Train RVC model error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Enhanced Speaker Diarization (pyannote.audio) ====================

@router.post("/diarization/enhanced", response_model=ApiResponse[dict])
async def enhanced_speaker_diarization(
    audio_file: UploadFile = File(..., description="Audio file to analyze"),
    min_speakers: int = Form(1, description="Minimum number of speakers"),
    max_speakers: int = Form(10, description="Maximum number of speakers"),
    backend: str = Form("pyannote", description="Backend: pyannote, speechbrain, modelscope, basic"),
    huggingface_token: Optional[str] = Form(None, description="HuggingFace token (for pyannote)"),
):
    """
    Perform enhanced speaker diarization using pyannote.audio or other backends.

    Backends:
    - pyannote: Professional-grade (requires HuggingFace token)
    - speechbrain: Research-grade ECAPA
    - modelscope: ModelScope/FunASR bundle (real model, large download on first use)
    - basic: Energy-based fallback (no dependencies)

    Returns speaker segments with timing and confidence.
    """
    try:
        # Save audio file
        temp_dir = Path(tempfile.gettempdir()) / "enhanced_diarization"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path = temp_dir / audio_file.filename
        with audio_path.open("wb") as f:
            content = await audio_file.read()
            f.write(content)

        try:
            # For model-based diarization backends, use WAV for maximum decode compatibility.
            effective_audio_path = audio_path
            converted_path: Path | None = None
            if audio_path.suffix.lower() != ".wav" and backend in ("pyannote", "speechbrain", "modelscope"):
                from app.utils.audio_decode import ffmpeg_available

                if ffmpeg_available():
                    converted_path = audio_path.with_suffix(".wav")
                    import subprocess

                    subprocess.check_call(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(audio_path),
                            str(converted_path),
                        ]
                    )
                    effective_audio_path = converted_path

            # Parse backend
            try:
                diarization_backend = DiarizationBackend(backend)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend: {backend}"
                )

            # Run diarization
            diarization_service = get_enhanced_diarization_service(
                backend=diarization_backend,
                huggingface_token=huggingface_token,
            )

            result = await diarization_service.diarize(
                str(effective_audio_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            return ApiResponse(data=result.to_dict())

        finally:
            # Clean up
            if audio_path.exists():
                audio_path.unlink()
            if converted_path and converted_path.exists():
                converted_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced diarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diarization/embeddings", response_model=ApiResponse[dict])
async def extract_speaker_embeddings(
    audio_file: UploadFile = File(..., description="Audio file"),
    min_speakers: int = Form(1, description="Minimum speakers"),
    max_speakers: int = Form(5, description="Maximum speakers"),
):
    """
    Extract speaker embeddings for each detected speaker.

    Useful for speaker identification and matching.
    """
    try:
        # Save audio file
        temp_dir = Path(tempfile.gettempdir()) / "speaker_embeddings"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path = temp_dir / audio_file.filename
        with audio_path.open("wb") as f:
            content = await audio_file.read()
            f.write(content)

        try:
            # Get diarization service
            diarization_service = get_enhanced_diarization_service()

            # First, diarize to get segments
            diarization_result = await diarization_service.diarize(
                str(audio_path),
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )

            # Extract embeddings
            embeddings = await diarization_service.extract_speaker_embeddings(
                str(audio_path),
                diarization_result,
            )

            return ApiResponse(data={
                "speakers": list(embeddings.keys()),
                "embeddings_shape": {
                    speaker: emb.shape.tolist() if hasattr(emb, 'shape') else len(emb)
                    for speaker, emb in embeddings.items()
                },
                "num_speakers": len(embeddings),
            })

        finally:
            # Clean up
            if audio_path.exists():
                audio_path.unlink()

    except Exception as e:
        logger.error(f"Extract embeddings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/diarization/compare-speakers", response_model=ApiResponse[dict])
async def compare_speakers(
    audio1: UploadFile = File(..., description="First audio file"),
    audio2: UploadFile = File(..., description="Second audio file"),
):
    """
    Compare two audio files to determine if they're from the same speaker.

    Returns similarity score and confidence.
    """
    try:
        # Save audio files
        temp_dir = Path(tempfile.gettempdir()) / "speaker_comparison"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio1_path = temp_dir / audio1.filename
        audio2_path = temp_dir / audio2.filename

        with audio1_path.open("wb") as f:
            content = await audio1.read()
            f.write(content)

        with audio2_path.open("wb") as f:
            content = await audio2.read()
            f.write(content)

        try:
            # Get service and compare
            diarization_service = get_enhanced_diarization_service()
            result = await diarization_service.compare_speakers(
                str(audio1_path),
                str(audio2_path),
            )

            return ApiResponse(data=result)

        finally:
            # Clean up
            audio1_path.unlink(missing_ok=True)
            audio2_path.unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Compare speakers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diarization/backends", response_model=ApiResponse[dict])
async def get_diarization_backends():
    """
    Get list of supported diarization backends and their availability.
    """
    try:
        # Create temporary service to check backend availability
        from app.services.enhanced_speaker_diarization import EnhancedSpeakerDiarizationService
        temp_service = EnhancedSpeakerDiarizationService()

        backends = temp_service.get_supported_backends()

        return ApiResponse(data={
            "backends": backends,
        })

    except Exception as e:
        logger.error(f"Get backends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/diarization/install/{backend}", response_model=ApiResponse[dict])
async def get_installation_instructions(backend: str):
    """
    Get installation instructions for a diarization backend.
    """
    try:
        diarization_service = get_enhanced_diarization_service()
        instructions = await diarization_service.get_installation_instructions(backend)

        return ApiResponse(data={
            "backend": backend,
            "instructions": instructions,
        })

    except Exception as e:
        logger.error(f"Get installation instructions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ASR Speech-to-Text ====================

@router.post("/asr/transcribe", response_model=ApiResponse[dict])
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code (e.g., zh, en, ja)"),
    task: str = Form("transcribe", description="Task: transcribe or translate"),
    timestamps: str = Form("segment", description="Timestamp level: segment, word, none"),
    backend: str = Form("faster_whisper", description="ASR backend: faster_whisper, whisper, groq, azure, google"),
    vad_filter: bool = Form(True, description="Apply voice activity detection"),
    api_key: Optional[str] = Form(None, description="API key for cloud services"),
):
    """
    Transcribe audio file to text using ASR.

    Supported backends:
    - faster_whisper: Optimized Whisper (recommended, 5-8x faster)
    - whisper: OpenAI Whisper (original)
    - groq: Groq Whisper API (ultra-fast, requires API key)
    - azure: Azure Speech Service
    - google: Google Cloud Speech-to-Text

    Returns:
    - Transcribed text
    - Detected language
    - Confidence score
    - Timestamps for segments/words
    - Processing time
    """
    try:
        # Save audio file
        temp_dir = Path(tempfile.gettempdir()) / "asr_transcription"
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path = temp_dir / audio_file.filename
        with audio_path.open("wb") as f:
            content = await audio_file.read()
            f.write(content)

        try:
            # Parse backend
            try:
                asr_backend = ASRBackend(backend)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend: {backend}"
                )

            # Get ASR service
            asr_service = get_asr_service(
                backend=asr_backend,
                api_key=api_key,
            )

            # Transcribe
            result = await asr_service.transcribe(
                str(audio_path),
                language=language,
                task=task,
                timestamps=timestamps,
                vad_filter=vad_filter,
            )

            return ApiResponse(data=result.to_dict())

        finally:
            # Clean up
            if audio_path.exists():
                audio_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/asr/batch", response_model=ApiResponse[dict])
async def transcribe_batch(
    audio_files: List[UploadFile] = File(..., description="Multiple audio files"),
    language: Optional[str] = Form(None, description="Language code"),
    backend: str = Form("faster_whisper", description="ASR backend"),
):
    """
    Transcribe multiple audio files in batch.

    Processes files concurrently for faster throughput.
    """
    try:
        # Save audio files
        temp_dir = Path(tempfile.gettempdir()) / "asr_batch"
        temp_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        temp_paths = []

        for file in audio_files:
            temp_path = temp_dir / file.filename
            temp_paths.append(temp_path)
            with temp_path.open("wb") as f:
                content = await file.read()
                f.write(content)
            paths.append(str(temp_path))

        try:
            # Parse backend
            try:
                asr_backend = ASRBackend(backend)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend: {backend}"
                )

            # Get service and transcribe
            asr_service = get_asr_service(backend=asr_backend)
            results = await asr_service.transcribe_batch(
                paths,
                language=language,
            )

            return ApiResponse(data={
                "total_files": len(results),
                "results": [r.to_dict() for r in results],
            })

        finally:
            # Clean up
            for temp_path in temp_paths:
                if temp_path.exists():
                    temp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch ASR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/asr/languages", response_model=ApiResponse[dict])
async def get_asr_languages():
    """
    Get list of supported languages for ASR.
    """
    try:
        asr_service = get_asr_service()
        languages = await asr_service.get_supported_languages()

        return ApiResponse(data={
            "languages": languages,
        })

    except Exception as e:
        logger.error(f"Get ASR languages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/asr/backends", response_model=ApiResponse[dict])
async def get_asr_backends():
    """
    Get list of supported ASR backends and their availability.
    """
    try:
        # Create temporary service to check backend availability
        from app.services.asr_service import ASRService
        temp_service = ASRService()

        backends = temp_service.get_supported_backends()

        return ApiResponse(data={
            "backends": backends,
        })

    except Exception as e:
        logger.error(f"Get ASR backends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/asr/install/{backend}", response_model=ApiResponse[dict])
async def get_asr_installation_instructions(backend: str):
    """
    Get installation instructions for an ASR backend.
    """
    try:
        asr_service = get_asr_service()
        instructions = await asr_service.get_installation_instructions(backend)

        return ApiResponse(data={
            "backend": backend,
            "instructions": instructions,
        })

    except Exception as e:
        logger.error(f"Get ASR installation instructions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/asr/transcribe-url", response_model=ApiResponse[dict])
async def transcribe_from_url(
    audio_url: str = Form(..., description="URL of audio file"),
    language: Optional[str] = Form(None, description="Language code"),
    backend: str = Form("faster_whisper", description="ASR backend"),
):
    """
    Transcribe audio from URL.

    Downloads audio from URL and transcribes it.
    """
    try:
        import aiohttp

        # Download audio
        temp_dir = Path(tempfile.gettempdir()) / "asr_url"
        temp_dir.mkdir(parents=True, exist_ok=True)

        filename = audio_url.split("/")[-1].split("?")[0]
        audio_path = temp_dir / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(audio_url) as response:
                if response.status != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download audio: {response.status}"
                    )
                with audio_path.open("wb") as f:
                    f.write(await response.read())

        try:
            # Parse backend
            try:
                asr_backend = ASRBackend(backend)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid backend: {backend}"
                )

            # Get service and transcribe
            asr_service = get_asr_service(backend=asr_backend)
            result = await asr_service.transcribe(
                str(audio_path),
                language=language,
            )

            return ApiResponse(data=result.to_dict())

        finally:
            # Clean up
            if audio_path.exists():
                audio_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR from URL error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
