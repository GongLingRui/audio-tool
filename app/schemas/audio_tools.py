"""Audio tools schemas for voice studio hub."""
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Audio Quality Detection Schemas
# =============================================================================

class AudioQualityCheckRequest(BaseModel):
    """Audio quality check request."""

    file_path: str
    include_recommendations: bool = True


class AudioQualityResult(BaseModel):
    """Audio quality check result."""

    overall_score: int = Field(..., ge=0, le=100, description="Overall quality score 0-100")
    duration: float = Field(..., description="Audio duration in seconds")
    format: str = Field(..., description="Audio format (WAV, MP3, etc.)")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    channels: int = Field(..., ge=1, le=8, description="Number of audio channels")
    bit_depth: int = Field(..., description="Bit depth (16, 24, 32)")
    loudness: float = Field(..., description="Loudness in LUFS")
    dynamic_range: float = Field(..., description="Dynamic range in dB")
    recommendations: list[str] = Field(default_factory=list, description="Quality improvement recommendations")


# =============================================================================
# ASR (Automatic Speech Recognition) Schemas
# =============================================================================

class ASRSegment(BaseModel):
    """ASR time-aligned segment."""

    start: float = Field(..., ge=0, description="Segment start time in seconds")
    end: float = Field(..., ge=0, description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text")


class ASRRequest(BaseModel):
    """ASR transcription request."""

    file_path: str
    engine: str = Field(default="faster_whisper", description="ASR engine to use")
    language: str = Field(default="zh", description="Language code (zh, en, ja, ko, auto)")
    word_timestamps: bool = False


class ASRResult(BaseModel):
    """ASR transcription result."""

    text: str = Field(..., description="Full transcribed text")
    language: str = Field(..., description="Detected/specified language")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    segments: list[ASRSegment] = Field(default_factory=list, description="Time-aligned segments")


# =============================================================================
# Speaker Diarization Schemas
# =============================================================================

class DiarizationSegment(BaseModel):
    """Speaker diarization segment."""

    speaker: str = Field(..., description="Speaker identifier (e.g., SPEAKER_1)")
    start: float = Field(..., ge=0, description="Segment start time in seconds")
    end: float = Field(..., ge=0, description="Segment end time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class DiarizationRequest(BaseModel):
    """Speaker diarization request."""

    file_path: str
    backend: str = Field(default="pyannote", description="Diarization backend")
    min_speakers: int = Field(default=2, ge=1, le=10, description="Minimum number of speakers")
    max_speakers: int = Field(default=4, ge=1, le=10, description="Maximum number of speakers")


class DiarizationResult(BaseModel):
    """Speaker diarization result."""

    segments: list[DiarizationSegment] = Field(default_factory=list)
    num_speakers: int = Field(..., description="Number of detected speakers")


# =============================================================================
# RVC (Voice Conversion) Schemas
# =============================================================================

class RVCModel(BaseModel):
    """RVC model information."""

    model_id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Model display name")
    language: str = Field(..., description="Model language (zh-CN, en, etc.)")
    status: str = Field(..., pattern="^(available|loading)$", description="Model status")
    description: str | None = None


class RVCConvertRequest(BaseModel):
    """RVC voice conversion request."""

    file_path: str
    model_id: str = Field(..., description="Target model ID")
    f0_method: str = Field(default="rmvpe", pattern="^(rmvpe|crepe|pm)$", description="F0 extraction method")
    pitch_change: int = Field(default=0, ge=-12, le=12, description="Pitch adjustment in semitones")


class RVCConvertResult(BaseModel):
    """RVC voice conversion result."""

    audio_url: str = Field(..., description="Converted audio file URL")
    duration: float = Field(..., description="Output audio duration")
    model_used: str = Field(..., description="Model used for conversion")


# =============================================================================
# Dialect (Language Detection and Conversion) Schemas
# =============================================================================

class DialectDetectRequest(BaseModel):
    """Language/dialect detection request."""

    text: str = Field(..., min_length=1, max_length=10000, description="Text to analyze")


class DialectDetectResult(BaseModel):
    """Language/dialect detection result."""

    language: str = Field(..., description="Detected language/dialect name")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    language_code: str | None = None


class DialectConvertRequest(BaseModel):
    """Dialect conversion request."""

    text: str = Field(..., min_length=1, max_length=10000, description="Source text")
    target_dialect: str = Field(..., pattern="^(cantonese|minnan|hakka)$", description="Target dialect")


class DialectConvertResult(BaseModel):
    """Dialect conversion result."""

    original_text: str = Field(..., description="Original input text")
    converted_text: str = Field(..., description="Converted text")
    target_dialect: str = Field(..., description="Target dialect code")


# =============================================================================
# Model Quantization Schemas
# =============================================================================

class QuantizationRequest(BaseModel):
    """Model quantization request."""

    file_path: str
    quant_type: str = Field(default="int8", pattern="^(int8|fp16|dynamic)$", description="Quantization type")


class QuantizationResult(BaseModel):
    """Model quantization result."""

    original_size: float = Field(..., description="Original model size in MB")
    quantized_size: float = Field(..., description="Quantized model size in MB")
    compression_ratio: float = Field(..., ge=1, description="Compression ratio")
    speedup: float = Field(..., ge=1, description="Estimated inference speedup")
    output_path: str = Field(..., description="Path to quantized model")


# =============================================================================
# Dashboard Schemas
# =============================================================================

class TaskStatus(str):
    """Task status enum."""

    DONE = "done"
    PROCESSING = "processing"
    ERROR = "error"


class DashboardTask(BaseModel):
    """Dashboard task entry."""

    id: str
    name: str = Field(..., description="Task/file name")
    type: str = Field(..., description="Task type (ASR, Diarization, etc.)")
    status: str = Field(..., pattern="^(done|processing|error)$", description="Task status")
    time: str = Field(..., description="Relative time string")


class DashboardData(BaseModel):
    """Dashboard summary data."""

    recent_tasks: list[DashboardTask] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict, description="Additional statistics")
