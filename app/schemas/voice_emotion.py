"""Voice emotion and style control schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List


class EmotionParameters(BaseModel):
    """Emotion control parameters for TTS generation."""

    # Primary emotions
    happiness: Optional[float] = Field(None, ge=0, le=1, description="Joy/excitement level (0-1)")
    sadness: Optional[float] = Field(None, ge=0, le=1, description="Sorrow/melancholy level (0-1)")
    anger: Optional[float] = Field(None, ge=0, le=1, description="Anger/frustration level (0-1)")
    fear: Optional[float] = Field(None, ge=0, le=1, description="Fear/anxiety level (0-1)")
    surprise: Optional[float] = Field(None, ge=0, le=1, description="Surprise/shock level (0-1)")
    neutral: Optional[float] = Field(None, ge=0, le=1, description="Calm/neutral level (0-1)")

    # Secondary modifiers
    energy: Optional[float] = Field(None, ge=0, le=2, description="Energy/intensity (0-2, 1=normal)")
    tempo: Optional[float] = Field(None, ge=0.5, le=2, description="Speaking rate (0.5x-2x)")
    pitch: Optional[float] = Field(None, ge=-12, le=12, description="Pitch shift in semitones")
    volume: Optional[float] = Field(None, ge=0, le=2, description="Volume boost (0-2, 1=normal)")


class VoiceStyle(BaseModel):
    """Voice style and character definition."""

    name: str
    description: Optional[str] = None

    # Voice characteristics
    gender: Optional[str] = Field(None, pattern="^(male|female|neutral)$")
    age_range: Optional[str] = None  # e.g., "young-adult", "middle-aged", "senior"

    # Voice quality
    timbre: Optional[str] = None  # e.g., "warm", "bright", "husky", "clear"
    resonance: Optional[str] = None  # e.g., "chesty", "head", "nasal"

    # Speech pattern
    delivery: Optional[str] = None  # e.g., "staccato", "legato", "measured"
    accent: Optional[str] = None  # e.g., "american", "british", "australian"


class EmotionPreset(BaseModel):
    """Pre-configured emotion preset."""

    id: str
    name: str
    description: Optional[str] = None
    emotion: EmotionParameters
    example_instruct: Optional[str] = None


class VoiceStylingRequest(BaseModel):
    """Request for voice-styled TTS generation."""

    text: str
    emotion: EmotionParameters
    style: Optional[VoiceStyle] = None
    preset_id: Optional[str] = None


class VoiceConversionRequest(BaseModel):
    """Request for voice-to-voice conversion."""

    source_audio_path: str
    target_voice_id: str
    preserve_timing: bool = True
    preserve_prosody: bool = False

    # Voice adjustment parameters
    pitch_shift: Optional[float] = Field(None, ge=-12, le=12, description="Pitch shift in semitones (-12 to +12)")
    speed_factor: Optional[float] = Field(None, ge=0.5, le=2.0, description="Speed multiplier (0.5x to 2x)")


class BatchVoiceCloneRequest(BaseModel):
    """Request for batch voice cloning."""

    voice_samples: List[str]  # List of audio file paths
    voice_name: str
    description: Optional[str] = None
    language: str = "zh"


class TTSLanguageConfig(BaseModel):
    """Language-specific TTS configuration."""

    language_code: str
    language_name: str
    supported_voices: List[str]
    emotion_support: bool
    sample_rate: int
    model_type: str  # e.g., "neural", "concatenative", "hybrid"


class SpeechEnhancementRequest(BaseModel):
    """Request for speech enhancement and post-processing."""

    audio_path: str
    enhance_denoise: bool = True
    enhance_volume: bool = True
    add_compression: bool = False
    target_lufs: Optional[float] = Field(-16.0, ge=-60, le=0)
