"""Audio task schemas for new audio processing features."""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AudioTaskType:
    """Audio task type constants."""
    VOICE_CONVERSION = "voice_conversion"
    AUDIO_ENHANCEMENT = "audio_enhancement"
    SPEAKER_DIARIZATION = "speaker_diarization"
    SUPER_RESOLUTION = "super_resolution"
    TTS_GENERATION = "tts_generation"


class AudioTaskStatus:
    """Task status constants."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AudioTaskBase(BaseModel):
    """Base audio task schema."""

    task_type: str = Field(..., description="Type of audio task")
    input_audio_path: str = Field(..., description="Path to input audio file")
    input_text: str | None = Field(None, description="Optional input text")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Task parameters")


class CreateAudioTaskRequest(AudioTaskBase):
    """Audio task creation request schema."""

    project_id: str = Field(..., description="Project ID")


class AudioTaskResponse(AudioTaskBase):
    """Audio task response schema."""

    id: str
    project_id: str
    status: str
    output_audio_path: str | None
    output_data: dict[str, Any] | None
    progress: float
    error_message: str | None
    processing_time: float | None
    quality_score: float | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None

    class Config:
        from_attributes = True


class AudioTaskStatus(BaseModel):
    """Audio task status schema."""

    id: str
    task_type: str
    status: str
    progress: float
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None


# Voice Conversion Schemas
class VoiceConversionRequest(BaseModel):
    """Voice conversion request schema."""

    source_audio_path: str = Field(..., description="Path to source audio")
    target_voice_id: str = Field(..., description="Target voice configuration ID")
    similarity: float = Field(default=0.85, ge=0.0, le=1.0, description="Similarity level")
    prosody_transfer: bool = Field(default=True, description="Transfer prosody features")


class VoiceConversionResponse(BaseModel):
    """Voice conversion response schema."""

    task_id: str
    output_audio_path: str | None
    quality_score: float | None


# Audio Enhancement Schemas
class AudioEnhancementRequest(BaseModel):
    """Audio enhancement request schema."""

    input_audio_path: str = Field(..., description="Path to input audio")
    denoise: bool = Field(default=True, description="Apply noise reduction")
    normalize: bool = Field(default=True, description="Normalize audio levels")
    remove_reverb: bool = Field(default=False, description="Remove reverb")
    eq_preset: str | None = Field(None, description="EQ preset to apply")


class AudioEnhancementResponse(BaseModel):
    """Audio enhancement response schema."""

    task_id: str
    output_audio_path: str | None
    metrics: dict[str, float] | None


class AudioQualityReport(BaseModel):
    """Audio quality analysis report schema."""

    overall_score: float = Field(..., ge=0.0, le=100.0)
    noise_level: float
    clarity: float
    dynamic_range: float | None
    snr_db: float | None
    recommendations: list[str]


class EnhancementPreset(BaseModel):
    """Audio enhancement preset schema."""

    id: str
    name: str
    description: str
    settings: dict[str, Any]


# Speaker Diarization Schemas
class SpeakerDiarizationRequest(BaseModel):
    """Speaker diarization request schema."""

    input_audio_path: str = Field(..., description="Path to input audio")
    min_speakers: int | None = Field(None, ge=1, description="Minimum number of speakers")
    max_speakers: int | None = Field(None, ge=1, description="Maximum number of speakers")


class SpeakerSegment(BaseModel):
    """Speaker segment schema."""

    start: float = Field(..., ge=0.0, description="Segment start time in seconds")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds")
    speaker: str = Field(..., description="Speaker identifier")


class SpeakerDiarizationResponse(BaseModel):
    """Speaker diarization response schema."""

    task_id: str
    segments: list[SpeakerSegment]
    num_speakers: int
    speakers: list[str]
    duration: float


class TranscriptionSegment(BaseModel):
    """Transcription segment schema."""

    start: float = Field(..., ge=0.0)
    end: float = Field(..., ge=0.0)
    speaker: str
    text: str


class TranscriptionResponse(BaseModel):
    """Transcription response schema."""

    task_id: str
    segments: list[TranscriptionSegment]
    full_text: str
    num_speakers: int


# Audio Super-Resolution Schemas
class SuperResolutionRequest(BaseModel):
    """Audio super-resolution request schema."""

    input_audio_path: str = Field(..., description="Path to input audio")
    target_sr: int = Field(default=48000, ge=22050, le=96000, description="Target sample rate")
    enhance: bool = Field(default=True, description="Apply quality enhancement")


class SuperResolutionResponse(BaseModel):
    """Audio super-resolution response schema."""

    task_id: str
    output_audio_path: str | None
    original_sr: int | None
    target_sr: int
    quality_score: float | None
