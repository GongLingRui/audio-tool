"""Voice schemas."""
from pydantic import BaseModel, Field


class Voice(BaseModel):
    """Voice schema."""

    id: str
    name: str
    gender: str | None = None  # male, female
    language: str | None = None


class VoiceConfig(BaseModel):
    """Voice configuration schema."""

    speaker: str = Field(..., max_length=100)
    voice_type: str = Field(..., pattern="^(custom|clone|lora|design)$")
    voice_name: str | None = Field(None, max_length=100)
    style: str | None = Field(None, max_length=255)
    ref_audio_path: str | None = None
    lora_model_path: str | None = None
    language: str = "zh-CN"


class VoiceConfigSet(BaseModel):
    """Set of voice configurations."""

    voices: list[VoiceConfig]


class VoicePreviewRequest(BaseModel):
    """Voice preview request."""

    text: str = Field(..., min_length=1)
    voice_type: str
    voice_name: str | None = None
    instruct: str | None = None


class VoicePreviewResponse(BaseModel):
    """Voice preview response."""

    audio_url: str
    duration: float


class VoiceDesignRequest(BaseModel):
    """Voice design request."""

    description: str = Field(..., min_length=1)
    gender: str | None = Field(None, pattern="^(male|female)$")
    age_range: str | None = None
    style: str | None = None


class VoiceDesignResponse(BaseModel):
    """Voice design response."""

    preview_url: str | None
    voice_id: str
    suggested_config: dict | None = None
    message: str | None = None


class VoicesListResponse(BaseModel):
    """Voices list response."""

    custom: list[Voice]
    lora: list[Voice]


class SpeakerParseResponse(BaseModel):
    """Speaker parse response."""

    speakers: list[str]
    total_entries: int
