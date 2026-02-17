"""
Voice Consistency Service
Ensures character-to-voice configuration consistency across audiobook projects
"""

import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.voice_config import VoiceConfig
from app.models.chunk import Chunk
from app.models.project import Project

logger = logging.getLogger(__name__)


class ConsistencyIssueType(Enum):
    """Types of consistency issues."""
    MISSING_VOICE_CONFIG = "missing_voice_config"
    DUPLICATE_VOICE_CONFIG = "duplicate_voice_config"
    INCONSISTENT_VOICE = "inconsistent_voice"
    MISSING_REFERENCE_AUDIO = "missing_reference_audio"
    INVALID_EMOTION_RANGE = "invalid_emotion_range"
    UNREFERENCED_SPEAKER = "unreferenced_speaker"


@dataclass
class ConsistencyIssue:
    """Represents a consistency issue."""
    type: ConsistencyIssueType
    severity: str  # 'error', 'warning', 'info'
    speaker: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class ConsistencyReport:
    """Report from consistency check."""
    is_valid: bool
    issues: List[ConsistencyIssue]
    speakers_found: Set[str]
    voice_configs_count: int
    warnings_count: int
    errors_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "is_valid": self.is_valid,
            "issues": [
                {
                    "type": issue.type.value,
                    "severity": issue.severity,
                    "speaker": issue.speaker,
                    "message": issue.message,
                    "suggestion": issue.suggestion,
                }
                for issue in self.issues
            ],
            "speakers_found": list(self.speakers_found),
            "voice_configs_count": self.voice_configs_count,
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
        }


class VoiceConsistencyService:
    """
    Service for maintaining voice configuration consistency across audiobook projects.

    Ensures:
    - Each speaker has exactly one voice configuration
    - Reference audio is provided when voice cloning is used
    - Emotion parameters are within valid ranges
    - No orphaned voice configs exist
    """

    # Valid emotion parameter ranges
    EMOTION_RANGES = {
        "happiness": (0, 1),
        "sadness": (0, 1),
        "anger": (0, 1),
        "fear": (0, 1),
        "surprise": (0, 1),
        "neutral": (0, 1),
        "energy": (0.5, 2.0),
        "tempo": (0.5, 2.0),
        "pitch": (-6, 6),
        "volume": (0, 2.0),
    }

    # Default narrator voice config
    DEFAULT_NARRATOR_CONFIG = {
        "speaker": "NARRATOR",
        "voice_id": None,
        "engine": "qwen-tts",
        "emotion": {
            "neutral": 0.8,
            "energy": 1.0,
            "tempo": 1.0,
        },
        "stability": 0.8,
    }

    @staticmethod
    async def get_project_speakers(db: AsyncSession, project_id: str) -> Set[str]:
        """Extract all unique speakers from project chunks."""
        result = await db.execute(
            select(Chunk.speaker)
            .where(Chunk.project_id == project_id)
            .distinct()
        )
        return {row[0] for row in result.all() if row[0]}

    @staticmethod
    async def get_voice_configs_map(db: AsyncSession, project_id: str) -> Dict[str, VoiceConfig]:
        """Get voice configs indexed by speaker name."""
        result = await db.execute(
            select(VoiceConfig)
            .where(VoiceConfig.project_id == project_id)
        )
        configs = result.scalars().all()
        return {config.speaker: config for config in configs}

    @classmethod
    def validate_emotion_parameters(cls, emotion: Dict) -> List[ConsistencyIssue]:
        """Validate emotion parameter ranges."""
        issues = []

        for key, value in emotion.items():
            if key in cls.EMOTION_RANGES and value is not None:
                min_val, max_val = cls.EMOTION_RANGES[key]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    issues.append(ConsistencyIssue(
                        type=ConsistencyIssueType.INVALID_EMOTION_RANGE,
                        severity="warning",
                        speaker="unknown",
                        message=f"Emotion parameter '{key}' value {value} is outside valid range [{min_val}, {max_val}]",
                        suggestion=f"Adjust '{key}' to be between {min_val} and {max_val}"
                    ))

        return issues

    @classmethod
    async def check_consistency(cls, db: AsyncSession, project_id: str) -> ConsistencyReport:
        """
        Perform comprehensive consistency check for a project.

        Returns a detailed report of any issues found.
        """
        issues = []

        # Get all speakers from chunks
        speakers = await cls.get_project_speakers(db, project_id)

        # Get existing voice configs
        voice_configs = await cls.get_voice_configs_map(db, project_id)

        # Check for missing voice configs
        for speaker in speakers:
            if speaker not in voice_configs:
                issues.append(ConsistencyIssue(
                    type=ConsistencyIssueType.MISSING_VOICE_CONFIG,
                    severity="error",
                    speaker=speaker,
                    message=f"Speaker '{speaker}' has no voice configuration",
                    suggestion=f"Create a voice configuration for '{speaker}'"
                ))

        # Check existing voice configs
        for speaker, config in voice_configs.items():
            # Check if speaker is referenced in chunks
            if speaker not in speakers:
                issues.append(ConsistencyIssue(
                    type=ConsistencyIssueType.UNREFERENCED_SPEAKER,
                    severity="info",
                    speaker=speaker,
                    message=f"Voice config for '{speaker}' exists but speaker not found in script",
                    suggestion=f"This config can be removed or speaker may be used in future chunks"
                ))

            # Check reference audio for voice cloning
            if config.engine == "voice-clone" and not config.reference_audio_path:
                issues.append(ConsistencyIssue(
                    type=ConsistencyIssueType.MISSING_REFERENCE_AUDIO,
                    severity="error",
                    speaker=speaker,
                    message=f"Voice cloning enabled but no reference audio provided for '{speaker}'",
                    suggestion=f"Upload reference audio for '{speaker}' or change engine"
                ))

            # Validate emotion parameters
            if config.emotion:
                emotion_issues = cls.validate_emotion_parameters(config.emotion)
                for issue in emotion_issues:
                    issue.speaker = speaker
                    issues.append(issue)

        # Check for duplicate voice configs (same speaker with multiple configs)
        # This should not happen with unique constraint but worth checking
        result = await db.execute(
            select(VoiceConfig.speaker)
            .where(VoiceConfig.project_id == project_id)
            .group_by(VoiceConfig.speaker)
            .having(select(func.count()) > 1)
        )
        duplicates = result.all()

        for (speaker,) in duplicates:
            issues.append(ConsistencyIssue(
                type=ConsistencyIssueType.DUPLICATE_VOICE_CONFIG,
                severity="error",
                speaker=speaker,
                message=f"Multiple voice configurations found for speaker '{speaker}'",
                suggestion=f"Remove duplicate configurations, keep only one"
            ))

        # Count errors and warnings
        errors_count = sum(1 for i in issues if i.severity == "error")
        warnings_count = sum(1 for i in issues if i.severity == "warning")

        return ConsistencyReport(
            is_valid=errors_count == 0,
            issues=issues,
            speakers_found=speakers,
            voice_configs_count=len(voice_configs),
            warnings_count=warnings_count,
            errors_count=errors_count,
        )

    @classmethod
    async def create_missing_configs(
        cls,
        db: AsyncSession,
        project_id: str,
        default_engine: str = "qwen-tts",
        use_defaults: bool = True,
    ) -> List[VoiceConfig]:
        """
        Create voice configurations for speakers that don't have one.

        Args:
            db: Database session
            project_id: Project ID
            default_engine: Default TTS engine to use
            use_defaults: If True, use default narrator config for missing speakers

        Returns:
            List of created voice configurations
        """
        speakers = await cls.get_project_speakers(db, project_id)
        voice_configs = await cls.get_voice_configs_map(db, project_id)
        created = []

        for speaker in speakers:
            if speaker not in voice_configs:
                # Create default config
                config = VoiceConfig(
                    project_id=project_id,
                    speaker=speaker,
                    voice_id=None,
                    engine=default_engine,
                    emotion=cls.DEFAULT_NARRATOR_CONFIG["emotion"].copy(),
                    stability=cls.DEFAULT_NARRATOR_CONFIG["stability"],
                )
                db.add(config)
                created.append(config)
                logger.info(f"Created default voice config for speaker: {speaker}")

        if created:
            await db.commit()

        return created

    @classmethod
    def normalize_speaker_name(cls, name: str) -> str:
        """
        Normalize speaker names for consistency.

        - Converts to uppercase
        - Removes extra whitespace
        - Standardizes common variations
        """
        if not name:
            return "NARRATOR"

        # Common variations
        variations = {
            "narrator": "NARRATOR",
            "旁白": "NARRATOR",
            "叙述者": "NARRATOR",
            "讲述者": "NARRATOR",
        }

        normalized = name.strip().upper()

        # Check variations
        lower = name.strip().lower()
        if lower in variations:
            normalized = variations[lower]

        return normalized

    @classmethod
    async def suggest_voice_config(
        cls,
        db: AsyncSession,
        project_id: str,
        speaker: str,
        analyze_text: bool = True,
    ) -> dict:
        """
        Suggest voice configuration for a speaker based on context.

        Analyzes the speaker's lines to suggest appropriate emotion and style.
        """
        # Get chunks for this speaker
        result = await db.execute(
            select(Chunk)
            .where(
                Chunk.project_id == project_id,
                Chunk.speaker == speaker,
            )
            .limit(20)  # Analyze first 20 chunks
        )
        chunks = result.scalars().all()

        if not chunks:
            # Return default config
            return cls.DEFAULT_NARRATOR_CONFIG.copy()

        # Analyze text content for emotion cues
        text_samples = [chunk.text for chunk in chunks]
        combined_text = " ".join(text_samples)

        # Simple keyword-based emotion detection
        emotion = {
            "neutral": 0.6,
            "energy": 1.0,
            "tempo": 1.0,
        }

        lower_text = combined_text.lower()

        # Detect emotion patterns
        if any(word in lower_text for word in ["激动", "兴奋", "开心", "快乐"]):
            emotion["happiness"] = 0.5
            emotion["energy"] = 1.2
        elif any(word in lower_text for word in ["悲伤", "难过", "痛苦", "哭泣"]):
            emotion["sadness"] = 0.6
            emotion["energy"] = 0.8
            emotion["tempo"] = 0.9
        elif any(word in lower_text for word in ["愤怒", "生气", "怒吼"]):
            emotion["anger"] = 0.6
            emotion["energy"] = 1.4
            emotion["volume"] = 1.2
        elif any(word in lower_text for word in ["紧张", "害怕", "恐惧"]):
            emotion["fear"] = 0.4
            emotion["energy"] = 1.1
            emotion["tempo"] = 1.1

        return {
            "speaker": speaker,
            "voice_id": None,
            "engine": "qwen-tts",
            "emotion": emotion,
            "stability": 0.8,
            "based_on_samples": len(chunks),
        }


# Singleton instance
_voice_consistency_service: Optional[VoiceConsistencyService] = None


def get_voice_consistency_service() -> VoiceConsistencyService:
    """Get the singleton voice consistency service instance."""
    global _voice_consistency_service
    if _voice_consistency_service is None:
        _voice_consistency_service = VoiceConsistencyService()
    return _voice_consistency_service
